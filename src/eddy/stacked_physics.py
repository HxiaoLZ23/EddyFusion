"""训练用多通道张量构建：物理场分离通道 + 可学习频域/多尺度支路可用的显式高频与粗尺度分量。

导出至 ``*.npy``（HWC float32），与 Ultralytics 约定一致：与 ``*.png`` 同 stem 时优先加载 ``.npy``。

**7 通道（推荐主实验，无 Mask）** — ``build_physics_stacked_hw7``：
    0-2 — BGR（V, U, ADT）；
    3 — 相对涡度 ζ；
    4 — Okubo–Weiss；
    5-6 — ADT 梯度分量 ∂ADT/∂x、∂ADT/∂y（分位归一化）。

**8 通道（历史/消融）** — ``build_physics_stacked_hw8``：
    0-2 — BGR；3 — ζ；4 — Lap(ADT)；5 — OW；6 — 粗尺度残差；7 — |∇ADT|。

**7ch 逐通道消融（``build_physics_stacked_ablation``）**：
    - ``6_no_ow`` — BGR + ζ + GradX + GradY（6 路）；
    - ``5_no_grad`` — BGR + ζ + OW（5 路，去掉 GradX/Y）；
    - ``6_no_zeta`` — BGR + OW + GradX + GradY（6 路）；
    - ``4_bgr_zeta`` — BGR + ζ（4 路）。

**均不含** OW 二值 Mask 或伪标签轮廓；Mask 仅作 YOLO-seg 监督，不作输入。
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


def _grad_xy(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(arr, dtype=np.float64)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
    return gx, gy


def _grad_mag(arr: np.ndarray) -> np.ndarray:
    gx, gy = _grad_xy(arr)
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


def _bgr_stack(adt: np.ndarray, u: np.ndarray, v: np.ndarray, *, p_lo: float, p_hi: float) -> np.ndarray:
    c0 = _norm01_quantile(adt, p_lo, p_hi)
    c1 = _norm01_quantile(u, p_lo, p_hi)
    c2 = _norm01_quantile(v, p_lo, p_hi)
    return np.stack([c2, c1, c0], axis=-1)


def build_physics_stacked_hw6_no_ow(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    zeta: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """6 通道：BGR + ζ + GradX + GradY（7ch 去掉 OW）。"""
    adt = np.asarray(adt, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    ch_bgr = _bgr_stack(adt, u, v, p_lo=p_lo, p_hi=p_hi)
    c3 = _norm01_quantile(zeta, p_lo, p_hi)
    gx, gy = _grad_xy(np.nan_to_num(adt))
    c4 = _norm01_quantile(gx, p_lo, p_hi)
    c5 = _norm01_quantile(gy, p_lo, p_hi)
    return np.concatenate([ch_bgr, np.stack([c3, c4, c5], axis=-1)], axis=-1).astype(np.float32)


def build_physics_stacked_hw5_no_grad(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    zeta: np.ndarray,
    ow: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """5 通道：BGR + ζ + OW（7ch 去掉 GradX/GradY 两路）。"""
    ch_bgr = _bgr_stack(adt, u, v, p_lo=p_lo, p_hi=p_hi)
    c3 = _norm01_quantile(np.asarray(zeta, dtype=np.float64), p_lo, p_hi)
    c4 = _norm01_quantile(np.asarray(ow, dtype=np.float64), p_lo, p_hi)
    return np.concatenate([ch_bgr, np.stack([c3, c4], axis=-1)], axis=-1).astype(np.float32)


def build_physics_stacked_hw6_no_zeta(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ow: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """6 通道：BGR + OW + GradX + GradY（7ch 去掉 ζ）。"""
    adt = np.asarray(adt, dtype=np.float64)
    ch_bgr = _bgr_stack(adt, u, v, p_lo=p_lo, p_hi=p_hi)
    c4 = _norm01_quantile(np.asarray(ow, dtype=np.float64), p_lo, p_hi)
    gx, gy = _grad_xy(np.nan_to_num(adt))
    c5 = _norm01_quantile(gx, p_lo, p_hi)
    c6 = _norm01_quantile(gy, p_lo, p_hi)
    return np.concatenate([ch_bgr, np.stack([c4, c5, c6], axis=-1)], axis=-1).astype(np.float32)


def build_physics_stacked_hw4_bgr_zeta(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    zeta: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """4 通道：BGR + ζ（3ch 基线 + 涡度一路）。"""
    ch_bgr = _bgr_stack(adt, u, v, p_lo=p_lo, p_hi=p_hi)
    c3 = _norm01_quantile(np.asarray(zeta, dtype=np.float64), p_lo, p_hi)
    return np.concatenate([ch_bgr, c3[..., None]], axis=-1).astype(np.float32)


def build_physics_stacked_hw4_bgr_ow(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ow: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """4 通道：BGR + OW（3ch 基线 + Okubo–Weiss 一路）。"""
    ch_bgr = _bgr_stack(adt, u, v, p_lo=p_lo, p_hi=p_hi)
    c3 = _norm01_quantile(np.asarray(ow, dtype=np.float64), p_lo, p_hi)
    return np.concatenate([ch_bgr, c3[..., None]], axis=-1).astype(np.float32)


def build_physics_stacked_hw5_bgr_grad(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """5 通道：BGR + GradX + GradY（3ch 基线 + ADT 梯度两路）。"""
    adt = np.asarray(adt, dtype=np.float64)
    ch_bgr = _bgr_stack(adt, u, v, p_lo=p_lo, p_hi=p_hi)
    gx, gy = _grad_xy(np.nan_to_num(adt))
    c3 = _norm01_quantile(gx, p_lo, p_hi)
    c4 = _norm01_quantile(gy, p_lo, p_hi)
    return np.concatenate([ch_bgr, np.stack([c3, c4], axis=-1)], axis=-1).astype(np.float32)


# 旧名笔误保留别名（实为 6 通道）
build_physics_stacked_hw5_no_ow = build_physics_stacked_hw6_no_ow

# 旧名别名（笔误时期）
build_physics_stacked_hw6_no_grad = build_physics_stacked_hw5_no_grad

ABLATION_PROFILES: dict[str, int] = {
    "7": 7,
    "8": 8,
    "4_bgr_zeta": 4,
    "4_bgr_ow": 4,
    "5_bgr_grad": 5,
    "5_no_grad": 5,
    "6_no_ow": 6,
    "6_no_grad": 5,  # 已废弃别名，=5_no_grad
    "6_no_zeta": 6,
}


def ablation_profile_channels(profile: str) -> int:
    key = str(profile).strip()
    if key not in ABLATION_PROFILES:
        raise ValueError(f"未知 stack_profile={profile!r}，可选 {sorted(ABLATION_PROFILES)}")
    return int(ABLATION_PROFILES[key])


def build_physics_stacked_ablation(
    profile: str,
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    zeta: np.ndarray,
    ow: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """按消融 profile 构建 HWC float32 张量。"""
    key = str(profile).strip()
    kw = dict(p_lo=p_lo, p_hi=p_hi)
    if key == "7":
        return build_physics_stacked_hw7(adt, u, v, zeta, ow, **kw)
    if key == "8":
        return build_physics_stacked_hw8(adt, u, v, zeta, ow, **kw)
    if key == "6_no_ow":
        return build_physics_stacked_hw6_no_ow(adt, u, v, zeta, **kw)
    if key in ("5_no_grad", "6_no_grad"):
        return build_physics_stacked_hw5_no_grad(adt, u, v, zeta, ow, **kw)
    if key == "6_no_zeta":
        return build_physics_stacked_hw6_no_zeta(adt, u, v, ow, **kw)
    if key == "4_bgr_zeta":
        return build_physics_stacked_hw4_bgr_zeta(adt, u, v, zeta, **kw)
    if key == "4_bgr_ow":
        return build_physics_stacked_hw4_bgr_ow(adt, u, v, ow, **kw)
    if key == "5_bgr_grad":
        return build_physics_stacked_hw5_bgr_grad(adt, u, v, **kw)
    raise ValueError(f"未知 stack_profile={profile!r}")


def build_physics_stacked_hw7(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    zeta: np.ndarray,
    ow: np.ndarray,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """7 通道物理增强：BGR + ζ + OW + GradX(ADT) + GradY(ADT)，无 Mask。"""
    adt = np.asarray(adt, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    ow = np.asarray(ow, dtype=np.float64)

    c0 = _norm01_quantile(adt, p_lo, p_hi)
    c1 = _norm01_quantile(u, p_lo, p_hi)
    c2 = _norm01_quantile(v, p_lo, p_hi)
    c3 = _norm01_quantile(zeta, p_lo, p_hi)
    c4 = _norm01_quantile(ow, p_lo, p_hi)
    gx, gy = _grad_xy(np.nan_to_num(adt))
    c5 = _norm01_quantile(gx, p_lo, p_hi)
    c6 = _norm01_quantile(gy, p_lo, p_hi)

    ch_bgr = np.stack([c2, c1, c0], axis=-1)
    return np.concatenate([ch_bgr, np.stack([c3, c4, c5, c6], axis=-1)], axis=-1).astype(np.float32)


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
