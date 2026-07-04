"""从 YOLO-seg 推理结果提取实例级几何属性（面积、质心、周长、类型等）。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# 与 eddy_dataset / eddy_yolo_export 类别约定一致
EDDY_CLASS_NAMES: dict[int, str] = {
    0: "eddy_cyclonic",
    1: "eddy_anticyclonic",
}
EDDY_TYPE_ZH: dict[int, str] = {
    0: "气旋涡（冷涡）",
    1: "反气旋涡（暖涡）",
}


def eddy_type_label(cls_id: int | None, *, zh: bool = False) -> str | None:
    if cls_id is None:
        return None
    mapping = EDDY_TYPE_ZH if zh else EDDY_CLASS_NAMES
    return mapping.get(int(cls_id))


def _contour_to_xy(cnt: np.ndarray) -> list[list[float]]:
    """Douglas-Peucker 简化轮廓，供前端 SVG 叠加。"""
    if cnt is None or len(cnt) == 0:
        return []
    peri = float(cv2.arcLength(cnt, True))
    eps = max(1.0, 0.002 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def _bbox_perimeter_xyxy(x1: float, y1: float, x2: float, y2: float) -> tuple[float, list[list[float]]]:
    """mask 缺失时用 bbox 近似周长与四边形轮廓。"""
    contour = [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]
    peri = 2.0 * (max(1.0, x2 - x1) + max(1.0, y2 - y1))
    return peri, contour


def geometry_to_stats_row(g: dict[str, Any], row_id: int) -> dict[str, Any]:
    """将单条 geometry 转为 API / 前端 stats_rows 条目。"""
    cls_id = g.get("class_id")
    if cls_id is not None:
        try:
            cls_id = int(cls_id)
        except (TypeError, ValueError):
            cls_id = None
    row: dict[str, Any] = {
        "id": int(row_id),
        "area_px": round(float(g.get("area_pixels") or 0), 2),
        "perimeter_px": round(float(g.get("perimeter_px") or 0), 2),
        "centroid_xy": list(g["centroid_xy"]) if g.get("centroid_xy") else [0.0, 0.0],
        "confidence": round(float(g.get("confidence") or 0), 4) if g.get("confidence") is not None else None,
        "class_id": cls_id,
        "eddy_type": g.get("eddy_type") or eddy_type_label(cls_id, zh=True),
        "contour_xy": g.get("contour_xy") or [],
    }
    if g.get("bbox_xyxy"):
        x1, y1, x2, y2 = g["bbox_xyxy"]
        row["bbox_xywh"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
    return row


def _mask_to_geometry(mask_u8: np.ndarray, conf: float, cls_id: int | None = None) -> dict[str, Any]:
    """mask_u8: 单通道 0/255，与图像同尺寸。"""
    h, w = mask_u8.shape[:2]
    m = (mask_u8 > 127).astype(np.uint8)
    cid = int(cls_id) if cls_id is not None else None
    if int(m.sum()) == 0:
        return {
            "area_pixels": 0.0,
            "perimeter_px": 0.0,
            "bbox_xyxy": None,
            "centroid_xy": None,
            "contour_xy": [],
            "angle_deg": None,
            "compactness": None,
            "confidence": float(conf),
            "class_id": cid,
            "eddy_type": eddy_type_label(cid, zh=True),
        }
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area_pixels": 0.0,
            "perimeter_px": 0.0,
            "bbox_xyxy": None,
            "centroid_xy": None,
            "contour_xy": [],
            "angle_deg": None,
            "compactness": None,
            "confidence": float(conf),
            "class_id": cid,
            "eddy_type": eddy_type_label(cid, zh=True),
        }
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if area < 1.0:
        return {
            "area_pixels": area,
            "perimeter_px": float(peri),
            "bbox_xyxy": None,
            "centroid_xy": None,
            "contour_xy": _contour_to_xy(cnt),
            "angle_deg": None,
            "compactness": None,
            "confidence": float(conf),
            "class_id": cid,
            "eddy_type": eddy_type_label(cid, zh=True),
        }
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
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
        "perimeter_px": float(peri),
        "bbox_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
        "min_rect_wh": (float(rw), float(rh)),
        "centroid_xy": centroid,
        "contour_xy": _contour_to_xy(cnt),
        "angle_deg": float(angle),
        "compactness": float(compact) if compact is not None else None,
        "confidence": float(conf),
        "class_id": cid,
        "eddy_type": eddy_type_label(cid, zh=True),
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
            peri, contour = _bbox_perimeter_xyxy(float(x1), float(y1), float(x2), float(y2))
            cid = int(clss[i])
            g = {
                "area_pixels": float(area),
                "area_frac": float(area / max(H * W, 1)),
                "perimeter_px": float(peri),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "centroid_xy": (float((x1 + x2) / 2), float((y1 + y2) / 2)),
                "contour_xy": contour,
                "angle_deg": 0.0,
                "compactness": None,
                "confidence": float(confs[i]),
                "class_id": cid,
                "eddy_type": eddy_type_label(cid, zh=True),
                "instance_id": i,
                "note": "mask 缺失，由 bbox 近似几何",
            }
            out.append(g)
    return out
