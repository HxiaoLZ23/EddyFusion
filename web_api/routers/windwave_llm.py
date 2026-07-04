from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from web_api.deps import resolve_nc_token

router = APIRouter(prefix="/windwave", tags=["windwave"])


class LlmReportBody(BaseModel):
    nc_path: str = Field(..., description="与 offline-report 相同的 NC 路径")
    model: str | None = Field(None, description="覆盖 DASHSCOPE_MODEL")
    max_tokens: int = 2048


def _linked_detect_from_nc(nc_path: Path) -> dict[str, Any]:
    from src.anomaly.detect import run_detect
    from src.anomaly.eddy_typhoon_bridge import (
        build_anomaly_result_for_detect,
        infer_typhoon_link_defaults_from_eddy_result,
    )
    from src.anomaly.windwave_nc_bridge import build_eddy_result_from_windwave_netcdf

    from web_api.deps import REPO_ROOT

    try:
        eddy_result = build_eddy_result_from_windwave_netcdf(nc_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"风浪 NC 解析失败: {e}") from e

    defaults = infer_typhoon_link_defaults_from_eddy_result(eddy_result)
    anomaly_result = build_anomaly_result_for_detect(eddy_result, link_defaults=defaults)
    ev = Path(str(defaults["events_json_path"]))
    if not ev.is_absolute():
        ev = (REPO_ROOT / ev).resolve()
    events_json_path = str(ev)
    auto_link = ev.is_file()
    try:
        linked = run_detect(
            anomaly_result=anomaly_result,
            auto_link_typhoon=auto_link,
            events_json_path=events_json_path,
            top_k=int(defaults["top_k"]),
        )
    except FileNotFoundError as e:
        if auto_link:
            linked = run_detect(anomaly_result=anomaly_result, auto_link_typhoon=False)
            linked["typhoon_link_note"] = f"台风索引不可用，已跳过联动: {e}"
        else:
            raise HTTPException(status_code=500, detail=str(e)) from e
    return linked


@router.post("/llm-report")
def windwave_llm_report(body: LlmReportBody) -> dict[str, Any]:
    """风浪 L1：服务端调用 DashScope 生成四段解读（密钥仅环境变量）。"""
    p = resolve_nc_token(body.nc_path)
    linked = _linked_detect_from_nc(p)

    from src.anomaly.llm_report import try_llm_report

    parsed, err, fp = try_llm_report(
        linked,
        model=(body.model or "").strip() or None,
        max_tokens=int(body.max_tokens),
    )
    if parsed is None:
        raise HTTPException(status_code=502, detail=err or "LLM 调用失败")
    return {
        "status": "success",
        "fingerprint": fp,
        "parsed": parsed,
        "summary_anomaly": parsed.get("summary_anomaly"),
        "impact": parsed.get("impact"),
        "historical_analogy": parsed.get("historical_analogy"),
        "actions": parsed.get("actions") or [],
    }
