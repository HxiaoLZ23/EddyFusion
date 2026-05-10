"""从 YOLO-seg 推理结果提取实例级几何属性（面积、质心、朝向等）。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _mask_to_geometry(mask_u8: np.ndarray, conf: float, cls_id: int | None = None) -> dict[str, Any]:
    """mask_u8: 单通道 0/255，与图像同尺寸。"""
    h, w = mask_u8.shape[:2]
    m = (mask_u8 > 127).astype(np.uint8)
    if int(m.sum()) == 0:
        return {
            "area_pixels": 0.0,
            "bbox_xyxy": None,
            "centroid_xy": None,
            "angle_deg": None,
            "compactness": None,
            "confidence": float(conf),
            "class_id": cls_id,
        }
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area_pixels": 0.0,
            "bbox_xyxy": None,
            "centroid_xy": None,
            "angle_deg": None,
            "compactness": None,
            "confidence": float(conf),
            "class_id": cls_id,
        }
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 1.0:
        return {
            "area_pixels": area,
            "bbox_xyxy": None,
            "centroid_xy": None,
            "angle_deg": None,
            "compactness": None,
            "confidence": float(conf),
            "class_id": cls_id,
        }
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    peri = float(cv2.arcLength(cnt, True))
    compact = (4.0 * math.pi * area / (peri * peri + 1e-6)) if peri > 0 else None
    M = cv2.moments(cnt)
    if abs(M["m00"]) > 1e-6:
        mc_x = M["m10"] / M["m00"]
        mc_y = M["m01"] / M["m00"]
        centroid = (float(mc_x), float(mc_y))
    else:
        centroid = (float(cx), float(cy))
    return {
        "area_pixels": area,
        "area_frac": float(area / max(h * w, 1)),
        "bbox_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
        "min_rect_wh": (float(rw), float(rh)),
        "centroid_xy": centroid,
        "angle_deg": float(angle),
        "compactness": float(compact) if compact is not None else None,
        "confidence": float(conf),
        "class_id": int(cls_id) if cls_id is not None else None,
    }


def geometries_from_ultralytics_result(pred: Any, orig_hw: tuple[int, int]) -> list[dict[str, Any]]:
    """
    pred: ultralytics.engine.results.Results（单张）
    orig_hw: (H, W) 原图尺寸，用于还原 mask 尺寸。
    """
    if pred is None:
        return []
    out: list[dict[str, Any]] = []
    H, W = orig_hw
    boxes = getattr(pred, "boxes", None)
    masks = getattr(pred, "masks", None)
    n = int(len(boxes)) if boxes is not None else 0
    if n == 0:
        return out

    confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes, "conf") else np.ones(n, dtype=np.float32)
    clss = boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.zeros(n, dtype=int)

    if masks is not None and getattr(masks, "data", None) is not None:
        md = masks.data.detach().cpu().numpy()
        mh, mw = md.shape[1], md.shape[2]
        for i in range(min(n, md.shape[0])):
            mi = md[i]
            mi = (mi * 255.0).astype(np.uint8) if mi.max() <= 1.0 else mi.astype(np.uint8)
            if (mh, mw) != (H, W):
                mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
            g = _mask_to_geometry(mi, float(confs[i]), int(clss[i]))
            g["instance_id"] = i
            out.append(g)
        return out

    if hasattr(boxes, "xyxy"):
        xyxy = boxes.xyxy.detach().cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            area = bw * bh
            g = {
                "area_pixels": float(area),
                "area_frac": float(area / max(H * W, 1)),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "centroid_xy": (float((x1 + x2) / 2), float((y1 + y2) / 2)),
                "angle_deg": 0.0,
                "compactness": None,
                "confidence": float(confs[i]),
                "class_id": int(clss[i]),
                "instance_id": i,
                "note": "mask 缺失，由 bbox 近似几何",
            }
            out.append(g)
    return out
