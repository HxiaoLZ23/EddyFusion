from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from web_api.deps import REPO_ROOT, resolve_nc_token

router = APIRouter(prefix="/windwave", tags=["windwave"])


class OfflineReportBody(BaseModel):
    nc_path: str = Field(..., description="白名单下的 NC 相对路径")


@router.post("/offline-report")
def offline_report(body: OfflineReportBody) -> dict[str, Any]:
    """从上传 NC 提取风浪时序 → run_detect → 规则文本报告（与 Streamlit 风浪页「从 NC 构建」一致）。"""
    p = resolve_nc_token(body.nc_path)

    from src.anomaly.detect import run_detect
    from src.anomaly.eddy_typhoon_bridge import (
        build_anomaly_result_for_detect,
        infer_typhoon_link_defaults_from_eddy_result,
    )
    from src.anomaly.report import render_report
    from src.anomaly.windwave_nc_bridge import build_eddy_result_from_windwave_netcdf

    try:
        eddy_result = build_eddy_result_from_windwave_netcdf(p)
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
            linked = run_detect(
                anomaly_result=anomaly_result,
                auto_link_typhoon=False,
            )
            linked["typhoon_link_note"] = f"台风索引不可用，已跳过联动: {e}"
        else:
            raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"run_detect 失败: {e}") from e

    text = render_report(detect_output=linked)
    ar = linked.get("anomaly_result") if isinstance(linked.get("anomaly_result"), dict) else {}
    return {
        "status": "success",
        "report_text": text,
        "anomaly_level": ar.get("anomaly_level"),
        "anomaly_index": ar.get("anomaly_index"),
        "typhoon_linked": bool((linked.get("typhoon_link") or {}).get("linked")),
        "typhoon_link_note": linked.get("typhoon_link_note"),
    }
