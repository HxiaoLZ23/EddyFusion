"""多尺度推理（轻量 TTA）：多 imgsz 预测后在框空间做 NMS 合并（分割掩码仍以主尺度绘制）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _boxes_from_result(res: Any) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    bx = getattr(res, "boxes", None)
    if bx is None or len(bx) == 0:
        return None, None
    xyxy = bx.xyxy.detach().cpu().numpy()
    conf = bx.conf.detach().cpu().numpy()
    return xyxy, conf


def multiscale_box_merge(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    base_imgsz: int = 640,
    scales: tuple[float, ...] = (0.9, 1.0, 1.1),
    conf: float = 0.25,
    iou: float = 0.5,
    device_hint: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    返回合并后的 xyxy[N,4], conf[N], best_imgsz（用于可选二次 predict 出图）。
    """
    xyxys = []
    confs = []
    best_sz = base_imgsz
    max_n = -1
    for s in scales:
        imgsz = max(32, int(round(base_imgsz * s)))
        kw: dict[str, Any] = {}
        if device_hint:
            kw["device"] = device_hint
        pred_list = model.predict(
            frame_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            **kw,
        )
        res = pred_list[0] if pred_list else None
        xy, cf = _boxes_from_result(res)
        if xy is not None:
            xyxys.append(xy)
            confs.append(cf)
            if len(xy) > max_n:
                max_n = len(xy)
                best_sz = imgsz
    if not xyxys:
        return None, None, base_imgsz
    xy_all = np.concatenate(xyxys, axis=0)
    cf_all = np.concatenate(confs, axis=0)
    order = np.argsort(-cf_all)
    xy_all = xy_all[order]
    cf_all = cf_all[order]
    try:
        from torchvision.ops import nms

        b = torch.from_numpy(xy_all).float()
        c = torch.from_numpy(cf_all).float()
        keep = nms(b, c, iou).detach().cpu().numpy()
    except Exception:
        keep = np.array(greedy_nms_numpy_indices(xy_all, cf_all, iou), dtype=np.int64)
    return xy_all[keep], cf_all[keep], best_sz


def greedy_nms_numpy_indices(xyxy: np.ndarray, conf: np.ndarray, iou_thresh: float) -> list[int]:
    if len(xyxy) == 0:
        return []
    order = np.argsort(-conf)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = bbox_iou_vector(xyxy[i], xyxy[rest])
        rest = rest[ious <= iou_thresh]
        order = rest
    return keep


def bbox_iou_vector(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    a = (box[2] - box[0]) * (box[3] - box[1])
    arest = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + arest - inter + 1e-9
    return inter / union


def tta_any_detection(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    base_imgsz: int = 640,
    scales: tuple[float, ...] = (0.85, 1.0, 1.15),
    conf: float = 0.25,
    iou: float = 0.45,
) -> bool:
    """任一分辨率上出现检测框则 True（用于前端轻量多尺度灵敏度提示）。"""
    for s in scales:
        imgsz = max(32, int(round(base_imgsz * s)))
        plist = model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        res = plist[0] if plist else None
        xy, _ = _boxes_from_result(res)
        if xy is not None and len(xy) > 0:
            return True
    return False
