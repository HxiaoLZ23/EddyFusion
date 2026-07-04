from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.eddy.geometry import _contour_to_xy, geometry_to_stats_row
from src.eddy.nc_to_bgr import extract_triple_scalar_fields_from_netcdf
from web_api.deps import resolve_eddy_preview_file, resolve_nc_token, resolve_repo_relative

router = APIRouter(prefix="/eddy", tags=["eddy"])

_PREVIEW_NAME = re.compile(r"^eddy_nc_(base|ann)_[A-Za-z0-9_-]{4,80}\.mp4$")


@router.get("/ping")
def eddy_ping() -> dict[str, object]:
    """不依赖 OpenCV；用于确认 `web_api` 已加载本路由（若 404 多为未重启或启动模块错误）。"""
    try:
        from src.eddy.mp4_browser_safe import mp4_encoder_status

        enc = mp4_encoder_status()
    except Exception as e:
        enc = {"error": str(e)}
    return {
        "ok": True,
        "dual_mp4": "POST /api/eddy/dual-mp4",
        "preview": "GET /api/eddy/preview?file=...",
        "mp4_encoder": enc,
    }


class DualMp4Request(BaseModel):
    nc_path: str = Field(..., description="白名单下的 NC 相对路径或绝对路径")
    channel_mode: str = Field("3ch", description="涡旋输入通道：论文演示仅支持 3ch（ADT+ADT+ADT，Fair-B0）")
    model_path: str | None = None
    conf: float = 0.25
    iou: float = 0.45
    base_imgsz: int = 640
    fps: float = 1.0
    max_frames: int = 120
    time_stride: int = 1
    time_start: int = 0
    time_stop: int | None = None
    #: full=一次返回双路；base=先底图（无 YOLO）；annotate=对已有 job 批量 YOLO 出带框路
    deliver: str = Field("full", description="full|base|annotate（staged 等同 full）")
    job_id: str | None = None


class DualMp4AnnotateBody(BaseModel):
    job_id: str
    fps: float = 1.0
    channel_mode: str = Field("3ch", description="涡旋输入通道：论文演示仅支持 3ch（ADT+ADT+ADT，Fair-B0）")
    model_path: str | None = None
    conf: float = 0.25
    iou: float = 0.45
    base_imgsz: int = 640


class EddyPreviewFrameBody(BaseModel):
    nc_path: str = Field(..., description="白名单下的 NC 相对路径或绝对路径")
    time_index: int = Field(0, ge=0, description="时间索引（从 0 开始）")


@router.post("/dual-mp4")
def build_dual_mp4(body: DualMp4Request) -> dict[str, Any]:
    try:
        from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path_for_stack
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"涡旋服务依赖未就绪（例如未安装 opencv-python-headless）: {e}",
        ) from e

    deliver = (body.deliver or "full").strip().lower()
    if deliver == "staged":
        deliver = "full"
    if deliver not in ("full", "base", "annotate"):
        raise HTTPException(status_code=400, detail="deliver 须为 full、base 或 annotate（staged 等同 full）")
    if deliver == "annotate":
        raise HTTPException(status_code=400, detail="annotate 请使用 POST /api/eddy/dual-mp4/annotate")

    nc = resolve_nc_token(body.nc_path)
    channel_mode = str(body.channel_mode or "3ch").strip().lower()
    if channel_mode != "3ch":
        raise HTTPException(status_code=400, detail="论文演示 API 仅支持 channel_mode=3ch（ADT+ADT+ADT）")
    if body.model_path and str(body.model_path).strip():
        mp = str(resolve_repo_relative(str(body.model_path).strip()))
    else:
        mp = default_eddy_weight_path_for_stack(channel_mode)  # type: ignore[arg-type]
    svc = EddyDemoService(
        model_path=mp,
        conf=float(body.conf),
        iou=float(body.iou),
        base_imgsz=int(body.base_imgsz),
    )
    out = svc.infer_netcdf_dual_mp4(
        nc_path=str(nc) if deliver != "annotate" else "",
        time_start=int(body.time_start),
        time_stop=body.time_stop,
        time_stride=int(body.time_stride),
        fps=float(body.fps),
        max_frames=int(body.max_frames),
        task_id=body.job_id,
        deliver=deliver,  # type: ignore[arg-type]
        job_id=body.job_id,
    )
    if out.get("status") != "success":
        raise HTTPException(status_code=500, detail=out.get("message", str(out)))
    resp = {**out}
    resp["channel_mode"] = channel_mode
    resp["model_path"] = mp
    if out.get("base_mp4"):
        resp["preview_base"] = Path(str(out["base_mp4"])).name
    if out.get("annotated_mp4"):
        resp["preview_annotated"] = Path(str(out["annotated_mp4"])).name
    return resp


@router.post("/dual-mp4/annotate")
def complete_dual_mp4(body: DualMp4AnnotateBody) -> dict[str, Any]:
    try:
        from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path_for_stack
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    channel_mode = str(body.channel_mode or "3ch").strip().lower()
    if channel_mode != "3ch":
        raise HTTPException(status_code=400, detail="论文演示 API 仅支持 channel_mode=3ch（ADT+ADT+ADT）")
    if body.model_path and str(body.model_path).strip():
        mp = str(resolve_repo_relative(str(body.model_path).strip()))
    else:
        mp = default_eddy_weight_path_for_stack(channel_mode)  # type: ignore[arg-type]
    svc = EddyDemoService(
        model_path=mp,
        conf=float(body.conf),
        iou=float(body.iou),
        base_imgsz=int(body.base_imgsz),
    )
    out = svc.infer_netcdf_dual_mp4(
        nc_path="",
        fps=float(body.fps),
        deliver="annotate",
        job_id=body.job_id,
    )
    if out.get("status") != "success":
        raise HTTPException(status_code=500, detail=out.get("message", str(out)))
    out["channel_mode"] = channel_mode
    out["model_path"] = mp
    return out


@router.get("/preview")
def stream_preview(file: str = Query(..., alias="file", description="eddy_nc_base_*.mp4 或 eddy_nc_ann_*.mp4 文件名")):
    if not _PREVIEW_NAME.match(file):
        raise HTTPException(status_code=400, detail="非法文件名")
    p = resolve_eddy_preview_file(file)
    return FileResponse(str(p), media_type="video/mp4", filename=file)


def _to_data_url_png(img_bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("PNG 编码失败")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@router.post("/preview-frame")
def preview_frame(body: EddyPreviewFrameBody) -> dict[str, Any]:
    """
    论文 Phase 2 帧级预览：调用 EddyDemoService 做真实 YOLO 单帧推理（3ch）。
    返回带检测框的 annotated PNG（data URL）+ stats_rows（质心/面积/周长/类型/轮廓）。
    若权重未就绪则回退到 ADT 阈值可视化，保证前端始终拿到帧图。
    """
    nc = resolve_nc_token(body.nc_path)

    # --- 尝试真实 YOLO 推理 ---
    try:
        from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path_for_stack

        mp = default_eddy_weight_path_for_stack("3ch")
        svc = EddyDemoService(model_path=mp)
        result = svc.infer_netcdf_frame(nc_path=str(nc), time_index=int(body.time_index))
    except Exception:
        result = None

    if result and result.get("status") == "success":
        annotated = result.get("annotated_frame_bgr")
        if annotated is None:
            annotated = result.get("base_frame_bgr")
        geoms: list[dict[str, Any]] = result.get("geometries") or []
        rows = [geometry_to_stats_row(g, i + 1) for i, g in enumerate(geoms[:30])]
        tl = result.get("timeline") or []
        n_det = int(tl[0]["count"]) if tl else len(rows)
        peak = float(tl[0]["score"]) if tl else 0.0
        time_label = (result.get("meta") or {}).get("time_label")
        source = "yolo"
    else:
        # 降级：ADT 阈值可视化
        try:
            from src.eddy.nc_to_bgr import extract_bgr_frame_from_netcdf

            annotated, meta_f = extract_bgr_frame_from_netcdf(str(nc), time_index=int(body.time_index))
            a, _, _, _ = extract_triple_scalar_fields_from_netcdf(str(nc), time_index=int(body.time_index))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"提取预览帧失败: {e}") from e
        a_f = np.asarray(a, dtype=np.float64)
        finite_m = np.isfinite(a_f)
        thr = float(np.percentile(a_f[finite_m], 88)) if finite_m.any() else 0.0
        mask = (a_f >= thr).astype(np.uint8)
        n_labels, labels, lbl_stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rows = []
        for i in range(1, n_labels):
            area = int(lbl_stats[i, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            x = int(lbl_stats[i, cv2.CC_STAT_LEFT])
            y = int(lbl_stats[i, cv2.CC_STAT_TOP])
            w = int(lbl_stats[i, cv2.CC_STAT_WIDTH])
            h = int(lbl_stats[i, cv2.CC_STAT_HEIGHT])
            blob = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            peri = float(cv2.arcLength(cnt, True)) if cnt is not None else 0.0
            contour_xy = _contour_to_xy(cnt) if cnt is not None else [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rows.append(
                {
                    "id": i,
                    "area_px": area,
                    "perimeter_px": round(peri, 2),
                    "bbox_xywh": [x, y, w, h],
                    "centroid_xy": [round(float(cent[i, 0]), 2), round(float(cent[i, 1]), 2)],
                    "confidence": None,
                    "class_id": None,
                    "eddy_type": None,
                    "contour_xy": contour_xy,
                }
            )
        rows = sorted(rows, key=lambda r: r["area_px"], reverse=True)[:30]
        n_det = len(rows)
        peak = 0.0
        time_label = meta_f.get("time_label")
        source = "adt_fallback"

    try:
        img_data_url = _to_data_url_png(annotated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "status": "ok",
        "source": source,
        "time_index": int(body.time_index),
        "time_label": time_label,
        "shape_hw": [int(annotated.shape[0]), int(annotated.shape[1])],
        "image_data_url": img_data_url,
        "stats_rows": rows,
        "summary": {
            "candidate_count": n_det,
            "peak_conf": round(peak, 4),
        },
    }
