from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from web_api.job_queue import (
    get_job,
    list_jobs,
    run_eddy_dual_mp4_job,
    run_windwave_forecast_job,
    submit_job,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])


class CreateJobBody(BaseModel):
    type: Literal["eddy_dual_mp4", "windwave_forecast"]
    nc_path: str
    fps: float = 1.0
    max_frames: int = 120
    time_stride: int = 1
    time_start: int = 0
    time_stop: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    channel_mode: str = "3ch"


@router.post("")
def create_job(body: CreateJobBody) -> dict[str, Any]:
    """提交异步任务，返回 job_id；轮询 GET /api/jobs/{id} 获取进度与结果。"""
    payload = body.model_dump()
    if body.type == "eddy_dual_mp4":
        job_id = submit_job("eddy_dual_mp4", payload, run_eddy_dual_mp4_job)
    elif body.type == "windwave_forecast":
        job_id = submit_job("windwave_forecast", payload, run_windwave_forecast_job)
    else:
        raise HTTPException(status_code=400, detail=f"未知任务类型: {body.type}")
    return {"status": "accepted", "job_id": job_id, "poll_url": f"/api/jobs/{job_id}"}


@router.get("")
def jobs_index(limit: int = 30) -> dict[str, Any]:
    return {"status": "ok", "jobs": list_jobs(limit=limit)}


@router.get("/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    rec = get_job(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    return {
        "status": "ok",
        "job": {
            "id": rec["id"],
            "type": rec["type"],
            "status": rec["status"],
            "progress": rec.get("progress", 0),
            "phase": rec.get("phase"),
            "message": rec.get("message"),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
            "result": rec.get("result"),
            "error": rec.get("error"),
        },
    }
