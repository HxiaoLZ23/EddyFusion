from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from web_api.deps import REPO_ROOT, resolve_nc_token

router = APIRouter(prefix="/windwave", tags=["windwave"])

DEFAULT_EVENTS_JSON = "data/processed/anomaly/typhoon_kb/events.json"


def _events_json_path() -> Path:
    return (REPO_ROOT / DEFAULT_EVENTS_JSON).resolve()


@router.get("/typhoon-kb/status")
def typhoon_kb_status() -> dict[str, Any]:
    """台风查询索引是否就绪（供前端展示与排错）。"""
    p = _events_json_path()
    ready = p.is_file()
    count = 0
    source = None
    if ready:
        try:
            import json

            rows = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                count = len(rows)
        except Exception:
            pass
        idx = (REPO_ROOT / "data/processed/anomaly/typhoon_index.json").resolve()
        if idx.is_file():
            try:
                import json

                meta = json.loads(idx.read_text(encoding="utf-8"))
                source = meta.get("source")
            except Exception:
                pass
    return {
        "ready": ready,
        "events_json_path": str(p),
        "events_count": count,
        "source": source,
        "seed_hint": "python scripts/seed_typhoon_kb_demo.py",
        "full_build_hint": "scripts/run_typhoon_kb.ps1",
    }


class NcPathBody(BaseModel):
    nc_path: str = Field(..., description="白名单下的 NC 相对路径")
    top_k: int = Field(default=5, ge=1, le=20, description="DTW Top-K 候选数")


def _run_windwave_detect_pipeline(nc_path: Path, *, top_k: int = 5) -> dict[str, Any]:
    """NC → 风浪时序 → run_detect（含 DTW Top-K）。"""
    from src.anomaly.detect import compute_series_anomaly_segments, run_detect
    from src.anomaly.eddy_typhoon_bridge import (
        build_anomaly_result_for_detect,
        infer_typhoon_link_defaults_from_eddy_result,
    )
    from src.anomaly.report import render_report
    from src.anomaly.windwave_nc_bridge import build_eddy_result_from_windwave_netcdf

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
            top_k=int(top_k),
        )
    except FileNotFoundError as e:
        if auto_link:
            linked = run_detect(anomaly_result=anomaly_result, auto_link_typhoon=False)
            linked["typhoon_link_note"] = f"台风索引不可用，已跳过联动: {e}"
        else:
            raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"run_detect 失败: {e}") from e

    ar = linked.get("anomaly_result") if isinstance(linked.get("anomaly_result"), dict) else {}
    tl = linked.get("typhoon_link") if isinstance(linked.get("typhoon_link"), dict) else {}
    level = str(ar.get("anomaly_level") or "unknown").lower()
    if auto_link and not tl.get("candidates"):
        q = tl.get("query") if isinstance(tl.get("query"), dict) else {}
        if level in ("medium", "high"):
            linked["typhoon_link_note"] = linked.get("typhoon_link_note") or "异常等级达橙/红，但未检索到 DTW 候选。"
        elif q:
            mode = str(q.get("history_search_mode") or "full")
            hist = f"{q.get('start_time')} ~ {q.get('end_time')}"
            linked["typhoon_link_note"] = linked.get("typhoon_link_note") or (
                f"历史检索（{mode}，{hist}）在海区 lon {q.get('lon_min')}~{q.get('lon_max')}、"
                f"lat {q.get('lat_min')}~{q.get('lat_max')} 内无轨迹交集。"
            )

    wind_obs = eddy_result.get("demo_wind_observed") if isinstance(eddy_result.get("demo_wind_observed"), list) else []
    wind_pred = eddy_result.get("demo_wind_predicted") if isinstance(eddy_result.get("demo_wind_predicted"), list) else []
    wave_obs = eddy_result.get("demo_wave_observed") if isinstance(eddy_result.get("demo_wave_observed"), list) else []
    wave_pred = eddy_result.get("demo_wave_predicted") if isinstance(eddy_result.get("demo_wave_predicted"), list) else []
    n_series = min(len(wind_obs), len(wind_pred), len(wave_obs), len(wave_pred))
    segments = compute_series_anomaly_segments(
        wind_observed=[float(x) for x in wind_obs[:n_series]],
        wind_predicted=[float(x) for x in wind_pred[:n_series]],
        wave_observed=[float(x) for x in wave_obs[:n_series]],
        wave_predicted=[float(x) for x in wave_pred[:n_series]],
    )
    meta = eddy_result.get("meta") if isinstance(eddy_result.get("meta"), dict) else {}
    extract_meta = meta.get("wind_wave_extract") if isinstance(meta.get("wind_wave_extract"), dict) else {}
    time_labels = extract_meta.get("time_labels")
    if not isinstance(time_labels, list):
        time_labels = [f"T+{i}" for i in range(n_series)]

    series = [
        {
            "step": i,
            "time": str(time_labels[i]) if i < len(time_labels) else f"T+{i}",
            "wind_observed": float(wind_obs[i]),
            "wind_predicted": float(wind_pred[i]),
            "wave_observed": float(wave_obs[i]),
            "wave_predicted": float(wave_pred[i]),
            "anomaly_index": segments[i]["anomaly_index"] if i < len(segments) else None,
            "level": segments[i]["level"] if i < len(segments) else "normal",
        }
        for i in range(n_series)
    ]

    pred_backend = eddy_result.get("prediction_backend")
    if not isinstance(pred_backend, str):
        pred_backend = (eddy_result.get("meta") or {}).get("prediction_backend")

    return {
        "eddy_result": eddy_result,
        "linked": linked,
        "report_text": render_report(detect_output=linked),
        "anomaly_result": ar,
        "typhoon_link": tl,
        "series": series,
        "anomaly_segments": segments,
        "events_json_path": events_json_path,
        "kb_ready": Path(events_json_path).is_file(),
        "prediction_backend": pred_backend,
        "prediction_meta": eddy_result.get("wind_wave_prediction_meta"),
    }


@router.post("/forecast")
def windwave_forecast(body: NcPathBody) -> dict[str, Any]:
    """
    论文 Phase 3：双头 LSTM 滑窗一步预测 + 逐时次异常分段 + DTW Top-K。
    观测来自 NC；预测默认 WindWaveLSTM（outputs/anomaly/best.pt），不足时降级平滑基线。
    """
    p = resolve_nc_token(body.nc_path)
    out = _run_windwave_detect_pipeline(p, top_k=int(body.top_k))
    ar = out["anomaly_result"]
    tl = out["typhoon_link"]
    candidates = tl.get("candidates") if isinstance(tl.get("candidates"), list) else []
    return {
        "status": "success",
        "nc_path": str(p),
        "times": [s["time"] for s in out["series"]],
        "wind_obs": [s["wind_observed"] for s in out["series"]],
        "wind_pred": [s["wind_predicted"] for s in out["series"]],
        "swh_obs": [s["wave_observed"] for s in out["series"]],
        "swh_pred": [s["wave_predicted"] for s in out["series"]],
        "series": out["series"],
        "anomaly_segments": out["anomaly_segments"],
        "anomaly_level": ar.get("anomaly_level"),
        "anomaly_index": ar.get("anomaly_index"),
        "assessment_note": ar.get("assessment_note"),
        "typhoon_linked": bool(tl.get("linked")),
        "typhoon_link_note": out["linked"].get("typhoon_link_note"),
        "typhoon_query": tl.get("query"),
        "typhoon_candidates": candidates[: int(body.top_k)],
        "typhoon_retrieval": tl.get("retrieval"),
        "typhoon_kb_ready": out["kb_ready"],
        "prediction_backend": out.get("prediction_backend"),
        "prediction_meta": out.get("prediction_meta"),
    }


@router.post("/offline-report")
def offline_report(body: NcPathBody) -> dict[str, Any]:
    """从上传 NC 提取风浪时序 → run_detect → 规则文本报告（与 Streamlit 风浪页「从 NC 构建」一致）。"""
    p = resolve_nc_token(body.nc_path)
    out = _run_windwave_detect_pipeline(p, top_k=int(body.top_k))
    ar = out["anomaly_result"]
    tl = out["typhoon_link"]
    candidates = tl.get("candidates") if isinstance(tl.get("candidates"), list) else []
    return {
        "status": "success",
        "report_text": out["report_text"],
        "anomaly_level": ar.get("anomaly_level"),
        "anomaly_index": ar.get("anomaly_index"),
        "wind_wave_series": out["series"],
        "typhoon_linked": bool(tl.get("linked")),
        "typhoon_link_note": out["linked"].get("typhoon_link_note"),
        "typhoon_query": tl.get("query"),
        "typhoon_candidates": candidates,
        "typhoon_events_path": tl.get("events_json_path") or out["events_json_path"],
        "typhoon_retrieval": tl.get("retrieval"),
        "typhoon_kb_ready": out["kb_ready"],
        "typhoon_kb_events_count": len(candidates) if out["kb_ready"] else 0,
        "prediction_backend": out.get("prediction_backend"),
        "prediction_meta": out.get("prediction_meta"),
    }
