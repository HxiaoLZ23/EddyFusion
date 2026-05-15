"""训练用多通道张量构建：物理场分离通道 + 可学习频域/多尺度支路可用的显式高频与粗尺度分量。

导出至 ``*.npy``（HWC float32），与 Ultralytics 约定一致：与 ``*.png`` 同 stem 时优先加载 ``.npy``。
通道语义（固定 8，与磁盘 PNG / OpenCV 一致）：
    0-2 — B,G,R 对应 Ultralytics/CV 的 **BGR**：即与 ``_rgb_from_fields`` → ``cv2.cvtColor(RGB2BGR)`` 后三通道顺序相同（BGR[0]=V, BGR[1]=U, BGR[2]=ADT）；
    3 — 相对涡度 zeta；
    4 — Laplacian(ADT)，鲁棒缩放；
    5 — Okubo–Weiss 标量；
    6 — 多尺度低频残差：`ADT - Upsample(GaussianBlur(Downsample(ADT)))`；
    7 — ADT 梯度幅 |∇ADT|。
"""

from __future__ import annotations

import cv2
import numpy as np


def _norm01_quantile(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    lo, hi = np.percentile(xf, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1e-9
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _laplacian2d(arr: np.ndarray) -> np.ndarray:
    """5 点离散 Laplacian（与 scipy 不强依赖）。"""
    a = np.asarray(arr, dtype=np.float64)
    z = np.zeros_like(a)
    z[1:-1, 1:-1] = (
        -4.0 * a[1:-1, 1:-1]
        + a[0:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, 0:-2]
        + a[1:-1, 2:]
    )
    return z


def _grad_mag(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def _coarse_highfreq(adt: np.ndarray) -> np.ndarray:
    """粗尺度高斯金字塔残差，作为多尺度专家输入之一。"""
    a = np.asarray(adt, dtype=np.float64)
    h, w = a.shape
    if h < 8 or w < 8:
        return np.zeros_like(a)

    small = cv2.resize(a, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), sigmaX=1.0)
    up = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
    return a - up


def relative_vorticity_and_okubo_weiss_from_uv(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """由同格点 u,v 用中心差分近似相对涡度 ζ=∂v/∂x−∂u/∂y 与 Okubo–Weiss 标量 W=Sn²+Ss²−ζ²。"""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    zeta = dv_dx - du_dy
    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    ow = sn * sn + ss * ss - zeta * zeta
    return zeta, ow


def build_physics_stacked_hw8(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    zeta: np.ndarray,
    ow: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """返回 float32 (H,W,8)，值域约在 [0,1]（除残差可为负并经 tanh-like 缩放）。"""
    adt = np.asarray(adt, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    ow = np.asarray(ow, dtype=np.float64)

    c0 = _norm01_quantile(adt, p_lo, p_hi)
    c1 = _norm01_quantile(u, p_lo, p_hi)
    c2 = _norm01_quantile(v, p_lo, p_hi)
    c3 = _norm01_quantile(zeta, p_lo, p_hi)

    lap = _laplacian2d(np.nan_to_num(adt))
    c4 = _norm01_quantile(lap, p_lo, p_hi)

    c5 = _norm01_quantile(ow, p_lo, p_hi)

    coarse_hf = _coarse_highfreq(np.nan_to_num(adt))
    c6 = _norm01_quantile(coarse_hf, p_lo, p_hi)

    gm = _grad_mag(np.nan_to_num(adt))
    c7 = _norm01_quantile(gm, p_lo, p_hi)

    # 与导出 PNG（RGB→imwrite BGR）在 cv2.imread/npy 直读时可比拟：前 3 维为 BGR
    ch_bgr = np.stack([c2, c1, c0], axis=-1)
    return np.concatenate([ch_bgr, np.stack([c3, c4, c5, c6, c7], axis=-1)], axis=-1).astype(
        np.float32
    )
