"""CuPy 版涡旋物理场（仅用于 benchmark / 探索；生产路径仍用 ``stacked_physics.py``）。"""

from __future__ import annotations

import numpy as np

from src.eddy.stacked_physics import build_physics_stacked_hw8, relative_vorticity_and_okubo_weiss_from_uv


def _require_cupy():
    from src.utils.cupy_bootstrap import bootstrap_cupy_dll_paths

    bootstrap_cupy_dll_paths()
    try:
        import cupy as cp  # type: ignore
    except ImportError as e:
        raise ImportError("请安装 cupy-cuda11x 或 cupy-cuda12x 以运行 GPU 对照") from e
    return cp


def relative_vorticity_and_okubo_weiss_from_uv_cupy(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cp = _require_cupy()
    u_g = cp.asarray(u, dtype=cp.float64)
    v_g = cp.asarray(v, dtype=cp.float64)
    du_dy, du_dx = cp.gradient(u_g)
    dv_dy, dv_dx = cp.gradient(v_g)
    zeta = dv_dx - du_dy
    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    ow = sn * sn + ss * ss - zeta * zeta
    return cp.asnumpy(zeta), cp.asnumpy(ow)


def _norm01_quantile_cupy(x, cp, p_lo: float, p_hi: float):
    xf = x[cp.isfinite(x)]
    if int(xf.size) == 0:
        return cp.zeros_like(x, dtype=cp.float32)
    lo, hi = cp.percentile(xf, (p_lo, p_hi))
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1e-9
    y = cp.clip((x - lo_f) / (hi_f - lo_f), 0.0, 1.0)
    return cp.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).astype(cp.float32)


def _laplacian2d_cupy(a, cp):
    z = cp.zeros_like(a)
    z[1:-1, 1:-1] = (
        -4.0 * a[1:-1, 1:-1]
        + a[0:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, 0:-2]
        + a[1:-1, 2:]
    )
    return z


def _grad_mag_cupy(a, cp):
    gx = cp.zeros_like(a)
    gy = cp.zeros_like(a)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
    return cp.sqrt(gx * gx + gy * gy + 1e-12)


def build_physics_stacked_hw8_cupy(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    zeta: np.ndarray | None = None,
    ow: np.ndarray | None = None,
    *,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    """
    尽量在 GPU 上完成向量化；``_coarse_highfreq`` 仍用 CPU OpenCV（与真实瓶颈拆分对照）。
    返回 numpy HWC float32。
    """
    import cv2

    cp = _require_cupy()
    adt_g = cp.asarray(adt, dtype=cp.float64)
    u_g = cp.asarray(u, dtype=cp.float64)
    v_g = cp.asarray(v, dtype=cp.float64)
    if zeta is None or ow is None:
        du_dy, du_dx = cp.gradient(u_g)
        dv_dy, dv_dx = cp.gradient(v_g)
        zeta_g = dv_dx - du_dy
        sn = du_dx - dv_dy
        ss = dv_dx + du_dy
        ow_g = sn * sn + ss * ss - zeta_g * zeta_g
    else:
        zeta_g = cp.asarray(zeta, dtype=cp.float64)
        ow_g = cp.asarray(ow, dtype=cp.float64)

    c0 = _norm01_quantile_cupy(adt_g, cp, p_lo, p_hi)
    c1 = _norm01_quantile_cupy(u_g, cp, p_lo, p_hi)
    c2 = _norm01_quantile_cupy(v_g, cp, p_lo, p_hi)
    c3 = _norm01_quantile_cupy(zeta_g, cp, p_lo, p_hi)
    lap = _laplacian2d_cupy(cp.nan_to_num(adt_g), cp)
    c4 = _norm01_quantile_cupy(lap, cp, p_lo, p_hi)
    c5 = _norm01_quantile_cupy(ow_g, cp, p_lo, p_hi)

    adt_cpu = cp.asnumpy(cp.nan_to_num(adt_g))
    h, w = adt_cpu.shape
    if h >= 8 and w >= 8:
        small = cv2.resize(adt_cpu, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (0, 0), sigmaX=1.0)
        up = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
        coarse_hf = adt_cpu - up
    else:
        coarse_hf = np.zeros_like(adt_cpu)
    c6 = _norm01_quantile_cupy(cp.asarray(coarse_hf, dtype=cp.float64), cp, p_lo, p_hi)

    gm = _grad_mag_cupy(cp.nan_to_num(adt_g), cp)
    c7 = _norm01_quantile_cupy(gm, cp, p_lo, p_hi)

    ch_bgr = cp.stack([c2, c1, c0], axis=-1)
    out = cp.concatenate([ch_bgr, cp.stack([c3, c4, c5, c6, c7], axis=-1)], axis=-1)
    return cp.asnumpy(out.astype(cp.float32))


def run_physics_cpu(adt: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u, v)
    return build_physics_stacked_hw8(adt, u, v, zeta, ow)


def run_physics_cupy(adt: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return build_physics_stacked_hw8_cupy(adt, u, v)
