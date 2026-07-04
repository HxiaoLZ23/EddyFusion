from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from web_api.deps import REPO_ROOT, resolve_nc_token
from web_api.routers.windwave_report import _run_windwave_detect_pipeline

router = APIRouter(prefix="/report", tags=["report"])

REPORTS_DIR = REPO_ROOT / "app" / "data" / "reports"


def _reports_dir() -> Path:
    d = REPORTS_DIR.resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_report_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告文件损坏: {e}") from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="报告文件格式无效")
    return data


class StructuredReportBody(BaseModel):
    nc_path: str = Field(..., description="白名单 NC 相对路径")
    top_k: int = Field(default=5, ge=1, le=20)
    format: Literal["markdown", "json"] = Field(default="markdown", description="导出格式")


@router.post("/structured")
def structured_report(body: StructuredReportBody) -> dict[str, Any]:
    """
    论文 §5.5.5：汇总异常判定、时空窗、DTW Top-K 与建议，导出结构化报告。
    """
    p = resolve_nc_token(body.nc_path)
    out = _run_windwave_detect_pipeline(p, top_k=int(body.top_k))
    ar = out["anomaly_result"]
    tl = out["typhoon_link"]
    query = tl.get("query") if isinstance(tl.get("query"), dict) else {}
    candidates = tl.get("candidates") if isinstance(tl.get("candidates"), list) else []

    fields: dict[str, Any] = {
        "nc_path": str(p),
        "anomaly_level": ar.get("anomaly_level"),
        "anomaly_index": ar.get("anomaly_index"),
        "wind_residual": ar.get("wind_residual"),
        "wave_residual": ar.get("wave_residual"),
        "wind_z": ar.get("wind_z"),
        "wave_z": ar.get("wave_z"),
        "threshold_rule": ar.get("threshold_rule", "3sigma"),
        "assessment_note": ar.get("assessment_note"),
        "time_window": {
            "start": query.get("start_time"),
            "end": query.get("end_time"),
        },
        "region": {
            "lon_min": query.get("lon_min"),
            "lon_max": query.get("lon_max"),
            "lat_min": query.get("lat_min"),
            "lat_max": query.get("lat_max"),
        },
        "typhoon_top_k": [
            {
                "rank": i + 1,
                "event_id": c.get("event_id") or c.get("id") or c.get("name"),
                "start_time": c.get("start_time"),
                "end_time": c.get("end_time"),
                "score": c.get("score"),
                "dtw_distance": c.get("dtw_distance"),
            }
            for i, c in enumerate(candidates[: int(body.top_k)])
        ],
        "series_len": len(out["series"]),
        "anomaly_segment_highlights": [
            s for s in out["anomaly_segments"] if s.get("level") in ("medium", "high")
        ][:12],
    }

    if body.format == "json":
        return {"status": "success", "format": "json", "fields": fields, "markdown": out["report_text"]}

    return {
        "status": "success",
        "format": "markdown",
        "markdown": out["report_text"],
        "fields": fields,
        "download_name": f"windwave_report_{Path(p).stem}.md",
    }


class SaveReportBody(BaseModel):
    nc_path: str = Field(..., description="关联 NC 相对路径")
    markdown: str = Field(..., description="报告正文（Markdown）")
    fields: dict[str, Any] | None = Field(None, description="结构化字段快照")
    source: Literal["windwave", "eddy", "combined"] = Field(default="windwave")
    mode: Literal["offline", "realtime"] = Field(default="offline")
    title: str | None = Field(None, description="可选标题")


@router.post("/save")
def save_report(body: SaveReportBody) -> dict[str, Any]:
    """将结构化报告写入 app/data/reports/，供报告管理页列表与再导出。"""
    nc = resolve_nc_token(body.nc_path)
    rid = uuid.uuid4().hex[:12]
    now = int(time.time())
    fields = body.fields if isinstance(body.fields, dict) else {}
    record: dict[str, Any] = {
        "id": rid,
        "created_at": now,
        "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "nc_path": str(nc.relative_to(REPO_ROOT.resolve()).as_posix()) if nc.is_relative_to(REPO_ROOT.resolve()) else str(nc),
        "source": body.source,
        "mode": body.mode,
        "title": body.title or f"风浪异常报告 · {Path(nc).name}",
        "anomaly_level": fields.get("anomaly_level"),
        "anomaly_index": fields.get("anomaly_index"),
        "markdown": body.markdown,
        "fields": fields,
    }
    out_path = _reports_dir() / f"{rid}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "success", "id": rid, "path": str(out_path.relative_to(REPO_ROOT.resolve()).as_posix())}


@router.get("/history")
def list_reports(limit: int = 50) -> dict[str, Any]:
    """列出已保存报告（按创建时间倒序）。"""
    rows: list[dict[str, Any]] = []
    for p in sorted(_reports_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = _load_report_file(p)
            rows.append(
                {
                    "id": data.get("id", p.stem),
                    "created_at": data.get("created_at"),
                    "created_at_iso": data.get("created_at_iso"),
                    "nc_path": data.get("nc_path"),
                    "source": data.get("source"),
                    "mode": data.get("mode"),
                    "title": data.get("title"),
                    "anomaly_level": data.get("anomaly_level"),
                    "anomaly_index": data.get("anomaly_index"),
                }
            )
        except HTTPException:
            continue
        if len(rows) >= max(1, min(int(limit), 200)):
            break
    return {"status": "success", "reports": rows, "count": len(rows)}


@router.get("/{report_id}")
def get_report(report_id: str) -> dict[str, Any]:
    """读取单条报告（含 Markdown 正文）。"""
    if not report_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="非法 report_id")
    p = _reports_dir() / f"{report_id}.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"报告不存在: {report_id}")
    data = _load_report_file(p)
    return {"status": "success", "report": data}
