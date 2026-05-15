from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from web_api.deps import resolve_eddy_preview_file, resolve_nc_token, resolve_repo_relative

router = APIRouter(prefix="/eddy", tags=["eddy"])

_PREVIEW_NAME = re.compile(r"^eddy_nc_(base|ann)_[A-Za-z0-9_-]{4,80}\.mp4$")


@router.get("/ping")
def eddy_ping() -> dict[str, str | bool]:
    """不依赖 OpenCV；用于确认 `web_api` 已加载本路由（若 404 多为未重启或启动模块错误）。"""
    return {"ok": True, "dual_mp4": "POST /api/eddy/dual-mp4", "preview": "GET /api/eddy/preview?file=..."}


class DualMp4Request(BaseModel):
    nc_path: str = Field(..., description="白名单下的 NC 相对路径或绝对路径")
    model_path: str | None = None
    conf: float = 0.25
    iou: float = 0.45
    base_imgsz: int = 640
    fps: float = 1.0
    max_frames: int = 120
    time_stride: int = 1
    time_start: int = 0
    time_stop: int | None = None


@router.post("/dual-mp4")
def build_dual_mp4(body: DualMp4Request) -> dict[str, Any]:
    try:
        from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"涡旋服务依赖未就绪（例如未安装 opencv-python-headless）: {e}",
        ) from e

    nc = resolve_nc_token(body.nc_path)
    if body.model_path and str(body.model_path).strip():
        mp = str(resolve_repo_relative(str(body.model_path).strip()))
    else:
        mp = default_eddy_weight_path()
    svc = EddyDemoService(
        model_path=mp,
        conf=float(body.conf),
        iou=float(body.iou),
        base_imgsz=int(body.base_imgsz),
    )
    out = svc.infer_netcdf_dual_mp4(
        nc_path=str(nc),
        time_start=int(body.time_start),
        time_stop=body.time_stop,
        time_stride=int(body.time_stride),
        fps=float(body.fps),
        max_frames=int(body.max_frames),
        task_id=None,
    )
    if out.get("status") != "success":
        raise HTTPException(status_code=500, detail=out.get("message", str(out)))
    base_name = Path(str(out["base_mp4"])).name
    ann_name = Path(str(out["annotated_mp4"])).name
    return {
        **out,
        "preview_base": base_name,
        "preview_annotated": ann_name,
    }


@router.get("/preview")
def stream_preview(file: str = Query(..., alias="file", description="eddy_nc_base_*.mp4 或 eddy_nc_ann_*.mp4 文件名")):
    if not _PREVIEW_NAME.match(file):
        raise HTTPException(status_code=400, detail="非法文件名")
    p = resolve_eddy_preview_file(file)
    return FileResponse(str(p), media_type="video/mp4", filename=file)
