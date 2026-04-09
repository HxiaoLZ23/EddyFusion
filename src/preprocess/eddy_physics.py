"""中尺度涡旋：由地转流分量计算相对涡度与 Okubo–Weiss 参数（平面近似，等经纬间距）。"""

from __future__ import annotations

import numpy as np


def _regular_spacing_deg(coords: np.ndarray) -> float:
    c = np.asarray(coords, dtype=np.float64).ravel()
    if c.size < 2:
        return 1.0
    d = np.diff(np.sort(c))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def velocity_gradients_m_s(
    u: np.ndarray,
    v: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    earth_radius_m: float = 6_371_000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    u,v: (H, W) 东向、北向地转流 (m/s)。
    lat_deg: (H,) 或 (H,1)；lon_deg: (W,) 或 (1,W)。
    返回 du_dx, du_dy, dv_dx, dv_dy，单位约 s^-1。
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    H, W = u.shape
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    if lat.size == H:
        lat2d = np.broadcast_to(lat[:, None], (H, W))
    elif lat.size == H * W:
        lat2d = lat.reshape(H, W)
    else:
        raise ValueError(f"lat 长度 {lat.size} 与网格 H={H} 不匹配")
    if lon.size == W:
        lon2d = np.broadcast_to(lon[None, :], (H, W))
    elif lon.size == H * W:
        lon2d = lon.reshape(H, W)
    else:
        raise ValueError(f"lon 长度 {lon.size} 与网格 W={W} 不匹配")

    dlat_rad = np.deg2rad(_regular_spacing_deg(lat))
    dlon_rad = np.deg2rad(_regular_spacing_deg(lon))
    dy = earth_radius_m * dlat_rad
    cos_lat = np.cos(np.deg2rad(lat2d))
    cos_lat = np.clip(cos_lat, 1e-6, None)
    dx = earth_radius_m * cos_lat * dlon_rad

    du_dy = np.gradient(u, axis=0) / dy
    du_dx = np.gradient(u, axis=1) / dx
    dv_dy = np.gradient(v, axis=0) / dy
    dv_dx = np.gradient(v, axis=1) / dx
    return du_dx, du_dy, dv_dx, dv_dy


def okubo_weiss_and_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ζ = ∂v/∂x − ∂u/∂y；Sn = ∂u/∂x − ∂v/∂y；Ss = ∂v/∂x + ∂u/∂y；W = Sn² + Ss² − ζ²。
    """
    du_dx, du_dy, dv_dx, dv_dy = velocity_gradients_m_s(u, v, lat_deg, lon_deg)
    zeta = dv_dx - du_dy
    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    ow = sn * sn + ss * ss - zeta * zeta
    return zeta, ow


def multi_percentile_vote_mask(
    ow: np.ndarray,
    percentiles: tuple[float, ...],
    *,
    min_votes: int,
) -> np.ndarray:
    """
    路径二（简化）：多个「W 低于分位数」的掩膜做投票，>= min_votes 为候选涡旋像素。
    OW 在涡旋核区常为相对低值（相对背景更「椭圆」）；用低分位数侧阈值。
    """
    ow = np.asarray(ow, dtype=np.float64)
    masks = []
    flat = ow[np.isfinite(ow)]
    if flat.size == 0:
        return np.zeros_like(ow, dtype=bool)
    for p in percentiles:
        thr = float(np.percentile(flat, p))
        masks.append(ow < thr)
    stack = np.stack(masks, axis=0)
    votes = stack.sum(axis=0)
    return votes >= int(min_votes)


def single_threshold_mask(ow: np.ndarray, percentile: float) -> np.ndarray:
    flat = ow[np.isfinite(ow)]
    if flat.size == 0:
        return np.zeros_like(ow, dtype=bool)
    thr = float(np.percentile(flat, percentile))
    return ow < thr
