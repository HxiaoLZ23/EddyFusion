"""全局异步任务队列（G16）：涡旋双 MP4、风浪预测等长耗时 API 后台执行。"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="api-job")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_MAX_JOBS = 200


def _now() -> float:
    return time.time()


def _prune_old() -> None:
    if len(_jobs) <= _MAX_JOBS:
        return
    done = [
        (jid, rec.get("updated_at", 0))
        for jid, rec in _jobs.items()
        if rec.get("status") in ("done", "failed")
    ]
    done.sort(key=lambda x: x[1])
    for jid, _ in done[: max(0, len(_jobs) - _MAX_JOBS)]:
        _jobs.pop(jid, None)


def _set(job_id: str, **fields: Any) -> None:
    with _lock:
        rec = _jobs.get(job_id)
        if not rec:
            return
        rec.update(fields)
        rec["updated_at"] = _now()


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        rec = _jobs.get(job_id)
        return dict(rec) if rec else None


def list_jobs(limit: int = 30) -> list[dict[str, Any]]:
    with _lock:
        rows = sorted(_jobs.values(), key=lambda r: r.get("created_at", 0), reverse=True)
    out: list[dict[str, Any]] = []
    for rec in rows[: max(1, min(int(limit), 100))]:
        out.append(
            {
                "id": rec["id"],
                "type": rec["type"],
                "status": rec["status"],
                "progress": rec.get("progress", 0),
                "phase": rec.get("phase"),
                "message": rec.get("message"),
                "created_at": rec.get("created_at"),
                "updated_at": rec.get("updated_at"),
            }
        )
    return out


def submit_job(job_type: str, payload: dict[str, Any], runner: Callable[[str, dict[str, Any]], None]) -> str:
    job_id = uuid.uuid4().hex[:12]
    record: dict[str, Any] = {
        "id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "phase": "queued",
        "message": "排队中",
        "created_at": _now(),
        "updated_at": _now(),
        "payload": payload,
        "result": None,
        "error": None,
    }
    with _lock:
        _jobs[job_id] = record
        _prune_old()

    def _wrap() -> None:
        _set(job_id, status="running", phase="start", progress=1, message="任务启动")
        try:
            runner(job_id, payload)
            _set(job_id, status="done", progress=100, phase="complete", message="完成")
        except Exception as e:
            _set(job_id, status="failed", phase="error", message=str(e), error=str(e))

    _executor.submit(_wrap)
    return job_id


def run_eddy_dual_mp4_job(job_id: str, payload: dict[str, Any]) -> None:
    from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path_for_stack
    from web_api.deps import resolve_nc_token

    nc = resolve_nc_token(str(payload["nc_path"]))
    channel_mode = str(payload.get("channel_mode") or "3ch").strip().lower()
    if channel_mode != "3ch":
        raise ValueError("仅支持 channel_mode=3ch")
    mp = default_eddy_weight_path_for_stack("3ch")
    svc = EddyDemoService(
        model_path=mp,
        conf=float(payload.get("conf", 0.25)),
        iou=float(payload.get("iou", 0.45)),
        base_imgsz=int(payload.get("base_imgsz", 640)),
    )

    _set(job_id, phase="extract", progress=10, message="规划时序并抽帧缓存…")
    base_out = svc.infer_netcdf_dual_mp4(
        nc_path=str(nc),
        time_start=int(payload.get("time_start", 0)),
        time_stop=payload.get("time_stop"),
        time_stride=int(payload.get("time_stride", 1)),
        fps=float(payload.get("fps", 1.0)),
        max_frames=int(payload.get("max_frames", 120)),
        deliver="base",
        job_id=job_id,
    )
    if base_out.get("status") != "success":
        raise RuntimeError(base_out.get("message", "底图阶段失败"))

    _set(job_id, phase="base_ready", progress=45, message="底图 MP4 已编码，YOLO 标注中…", result=base_out)
    ann_out = svc.infer_netcdf_dual_mp4(
        nc_path="",
        fps=float(payload.get("fps", 1.0)),
        deliver="annotate",
        job_id=base_out.get("job_id") or job_id,
    )
    if ann_out.get("status") != "success":
        raise RuntimeError(ann_out.get("message", "标注阶段失败"))

    merged = {**base_out, **ann_out, "phase": "complete"}
    if merged.get("base_mp4"):
        merged["preview_base"] = Path(str(merged["base_mp4"])).name
    if merged.get("annotated_mp4"):
        merged["preview_annotated"] = Path(str(merged["annotated_mp4"])).name
    _set(job_id, result=merged, progress=100, message="双路 MP4 完成")


def run_windwave_forecast_job(job_id: str, payload: dict[str, Any]) -> None:
    from web_api.deps import resolve_nc_token
    from web_api.routers.windwave_report import _run_windwave_detect_pipeline

    _set(job_id, phase="parse", progress=15, message="解析风浪 NC 时序…")
    nc = resolve_nc_token(str(payload["nc_path"]))
    top_k = int(payload.get("top_k", 5))

    _set(job_id, phase="detect", progress=40, message="WindWaveLSTM 推理与异常检测…")
    out = _run_windwave_detect_pipeline(nc, top_k=top_k)
    ar = out["anomaly_result"]
    tl = out["typhoon_link"]
    candidates = tl.get("candidates") if isinstance(tl.get("candidates"), list) else []

    result = {
        "status": "success",
        "nc_path": str(nc),
        "times": [s["time"] for s in out["series"]],
        "series": out["series"],
        "anomaly_segments": out["anomaly_segments"],
        "anomaly_level": ar.get("anomaly_level"),
        "anomaly_index": ar.get("anomaly_index"),
        "assessment_note": ar.get("assessment_note"),
        "typhoon_linked": bool(tl.get("linked")),
        "typhoon_link_note": out["linked"].get("typhoon_link_note"),
        "typhoon_query": tl.get("query"),
        "typhoon_candidates": candidates[:top_k],
        "typhoon_retrieval": tl.get("retrieval"),
        "typhoon_kb_ready": out["kb_ready"],
        "report_text": out["report_text"],
        "prediction_backend": out.get("prediction_backend"),
        "prediction_meta": out.get("prediction_meta"),
    }
    _set(job_id, phase="complete", progress=100, message="风浪预测完成", result=result)
