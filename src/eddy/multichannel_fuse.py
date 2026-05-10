"""将 SLA / 涡度 / 温度梯度等二维物理场融合为 YOLO 可用的三通道可视化输入。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _norm01(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(a[finite], (p_low, p_high))
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.float32)
    x = np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return x.astype(np.float32)


def fuse_physics_channels_to_bgr(
    sla: np.ndarray,
    vorticity: np.ndarray,
    temperature_gradient: np.ndarray,
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """
    三通道语义：B=SLA，G=相对涡度/涡度示意，R=温度水平梯度幅度（调用方传入已计算的标量场）。
    返回 uint8 HWC BGR，供 cv2/YOLO 使用。
    """
    for name, arr in ("sla", sla), ("vorticity", vorticity), ("temperature_gradient", temperature_gradient):
        if arr.ndim != 2:
            raise ValueError(f"{name} 须为二维 (H,W)，当前 ndim={arr.ndim}")
    h, w = sla.shape[:2]
    if vorticity.shape != (h, w) or temperature_gradient.shape != (h, w):
        raise ValueError("sla / vorticity / temperature_gradient 空间维度须一致")

    b = (_norm01(sla, p_low, p_high) * 255.0).astype(np.uint8)
    g = (_norm01(vorticity, p_low, p_high) * 255.0).astype(np.uint8)
    r = (_norm01(temperature_gradient, p_low, p_high) * 255.0).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def load_fused_bgr_from_npz(path: str | Path, *, keys: tuple[str, str, str] | None = None) -> np.ndarray:
    """
    NPZ 默认键：sla, vorticity, dtdy（temperature_gradient）。
    keys 可按顺序指定 (sla_key, vor_key, grad_key)。
    """
    path = Path(path)
    z = np.load(path)
    if keys is None:
        cand = (
            ("sla", "adt", "ssh"),
            ("vorticity", "vor", "rv", "relative_vorticity"),
            ("dtdy", "temp_grad", "temperature_gradient", "dt_dy"),
        )
        def pick(options: tuple[str, ...]) -> str:
            for k in options:
                if k in z.files:
                    return k
            raise KeyError(f"NPZ {path} 中未找到候选键之一: {options}")

        k0 = pick(cand[0])
        k1 = pick(cand[1])
        k2 = pick(cand[2])
    else:
        k0, k1, k2 = keys
        for k in keys:
            if k not in z.files:
                raise KeyError(f"NPZ 缺少键: {k}")
    sla = np.asarray(z[k0])
    vor = np.asarray(z[k1])
    tg = np.asarray(z[k2])
    if sla.ndim == 3:
        sla = sla[0]
    if vor.ndim == 3:
        vor = vor[0]
    if tg.ndim == 3:
        tg = tg[0]
    return fuse_physics_channels_to_bgr(sla, vor, tg)


def fused_bgr_from_arrays(arrs: dict[str, np.ndarray]) -> np.ndarray:
    """从已加载数组构造 BGR。"""
    return fuse_physics_channels_to_bgr(
        arrs["sla"],
        arrs["vorticity"],
        arrs["temperature_gradient"],
    )
