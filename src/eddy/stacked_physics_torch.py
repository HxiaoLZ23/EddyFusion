"""PyTorch CUDA 版涡旋物理场（benchmark 用；与 torch+cu118 共用运行时，无需单独装 CUDA Toolkit）。"""

from __future__ import annotations

import numpy as np
import torch

from src.eddy.stacked_physics import build_physics_stacked_hw8, relative_vorticity_and_okubo_weiss_from_uv


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def relative_vorticity_and_okubo_weiss_from_uv_torch(
    u: np.ndarray,
    v: np.ndarray,
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dev = device or _device()
    u_t = torch.as_tensor(u, dtype=torch.float64, device=dev)
    v_t = torch.as_tensor(v, dtype=torch.float64, device=dev)
    du_dy, du_dx = torch.gradient(u_t)
    dv_dy, dv_dx = torch.gradient(v_t)
    zeta = dv_dx - du_dy
    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    ow = sn * sn + ss * ss - zeta * zeta
    return zeta.detach().cpu().numpy(), ow.detach().cpu().numpy()


def _norm01_quantile_torch(x: torch.Tensor, p_lo: float, p_hi: float) -> torch.Tensor:
    xf = x[torch.isfinite(x)]
    if int(xf.numel()) == 0:
        return torch.zeros_like(x, dtype=torch.float32)
    lo = torch.quantile(xf, p_lo / 100.0)
    hi = torch.quantile(xf, p_hi / 100.0)
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1e-9
    y = torch.clamp((x - lo_f) / (hi_f - lo_f), 0.0, 1.0)
    return torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).to(torch.float32)


def _laplacian2d_torch(a: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(a)
    z[1:-1, 1:-1] = (
        -4.0 * a[1:-1, 1:-1]
        + a[0:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, 0:-2]
        + a[1:-1, 2:]
    )
    return z


def _grad_mag_torch(a: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(a)
    gy = torch.zeros_like(a)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def build_physics_stacked_hw8_torch(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    device: torch.device | None = None,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
) -> np.ndarray:
    import cv2

    dev = device or _device()
    adt_t = torch.as_tensor(adt, dtype=torch.float64, device=dev)
    u_t = torch.as_tensor(u, dtype=torch.float64, device=dev)
    v_t = torch.as_tensor(v, dtype=torch.float64, device=dev)
    du_dy, du_dx = torch.gradient(u_t)
    dv_dy, dv_dx = torch.gradient(v_t)
    zeta_t = dv_dx - du_dy
    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    ow_t = sn * sn + ss * ss - zeta_t * zeta_t

    c0 = _norm01_quantile_torch(adt_t, p_lo, p_hi)
    c1 = _norm01_quantile_torch(u_t, p_lo, p_hi)
    c2 = _norm01_quantile_torch(v_t, p_lo, p_hi)
    c3 = _norm01_quantile_torch(zeta_t, p_lo, p_hi)
    lap = _laplacian2d_torch(torch.nan_to_num(adt_t))
    c4 = _norm01_quantile_torch(lap, p_lo, p_hi)
    c5 = _norm01_quantile_torch(ow_t, p_lo, p_hi)

    adt_cpu = adt_t.detach().cpu().numpy()
    h, w = adt_cpu.shape
    if h >= 8 and w >= 8:
        small = cv2.resize(adt_cpu, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (0, 0), sigmaX=1.0)
        up = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
        coarse_hf = adt_cpu - up
    else:
        coarse_hf = np.zeros_like(adt_cpu)
    c6 = _norm01_quantile_torch(torch.as_tensor(coarse_hf, device=dev, dtype=torch.float64), p_lo, p_hi)
    gm = _grad_mag_torch(torch.nan_to_num(adt_t))
    c7 = _norm01_quantile_torch(gm, p_lo, p_hi)

    ch_bgr = torch.stack([c2, c1, c0], dim=-1)
    out = torch.cat([ch_bgr, torch.stack([c3, c4, c5, c6, c7], dim=-1)], dim=-1)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return out.detach().cpu().numpy().astype(np.float32)


def run_physics_torch(adt: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return build_physics_stacked_hw8_torch(adt, u, v)
