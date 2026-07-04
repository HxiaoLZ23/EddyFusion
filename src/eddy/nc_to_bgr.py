"""从命题方风格 NetCDF 抽取单帧 RGB→BGR，供 YOLO 涡旋检测（与 export_eddy_demo_video 变量约定一致）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.preprocess.netcdf_io import open_netcdf_dataset


def _pick_dataarray(ds: Any, names: tuple[str, ...]) -> Any:
    lower = {str(k).lower(): k for k in ds.data_vars}
    for n in names:
        if n.lower() in lower:
            return ds[lower[n.lower()]]
    raise KeyError(f"变量缺失，候选={names}，实际={list(ds.data_vars)}")


def _norm_to_u8(x: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.percentile(xf, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1e-9
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return (y * 255).astype(np.uint8)


def fair_adt_triple_bgr_u8(adt: np.ndarray, *, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Fair-B0 / V6 fair：单帧 ADT 经 P2/P98 拉伸后复制为三通道 BGR（与训练 export 一致）。"""
    ch = _norm_to_u8(adt, p_lo=p_lo, p_hi=p_hi)
    rgb = np.stack([ch, ch, ch], axis=-1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _spatial_only_vals(da: Any) -> np.ndarray:
    return np.asarray(da.values, dtype=np.float64)


def _isel_time(da: Any, tname: str, ti: int) -> np.ndarray:
    return np.asarray(da.isel({tname: ti}).values, dtype=np.float64)


def _time_dim_name(da: Any) -> str | None:
    sp = {"latitude", "longitude", "lat", "lon", "Latitude", "Longitude", "x", "y", "X", "Y"}
    for d in da.dims:
        if d not in sp and str(d).lower() not in ("x", "y"):
            return str(d)
    return None


def _time_label_for_index(da: Any, tname: str, ti: int) -> str | None:
    """从坐标读取人类可读时间戳（含 numpy datetime64、cftime、纳秒整数等）；失败则返回 None。"""
    try:
        if tname not in da.coords:
            return None
        vals = da.coords[tname].values
        if vals is None or int(ti) >= len(vals):
            return None
        v = vals[int(ti)]
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, np.datetime64):
            return str(np.datetime_as_string(v, unit="s"))
        if hasattr(v, "strftime"):
            try:
                return str(v.strftime("%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass
        try:
            import pandas as pd
        except ImportError:
            pd = None  # type: ignore[assignment]
        if pd is not None:
            try:
                ts = pd.Timestamp(v)
                if pd.notna(ts):
                    return ts.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            if isinstance(v, (int, float, np.floating, np.integer)):
                x = float(v)
                if abs(x) > 1e14:
                    try:
                        ts = pd.Timestamp(x, unit="ns")
                        if pd.notna(ts):
                            return ts.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
                if 1e11 < abs(x) < 1e14:
                    try:
                        ts = pd.Timestamp(x, unit="ms")
                        if pd.notna(ts):
                            return ts.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
        return str(v)
    except Exception:
        return None


def extract_triple_scalar_fields_from_netcdf(
    nc_path: str | Path,
    *,
    time_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    与 extract_bgr_frame_from_netcdf 相同的变量匹配逻辑，返回三通道**原始浮点场** (adt_like, u_like, v_like) 与 meta。
    用于由 SST/流场等构造 8 通道物理堆叠（eddy_enh）。
    """
    from src.eddy.nc_dual_batch import extract_triple_slices_batch

    rows = extract_triple_slices_batch(nc_path, [int(time_index)])
    if not rows:
        raise ValueError(f"无法从 NC 读取 time_index={time_index}")
    a0, u0, v0, meta = rows[0]
    return a0, u0, v0, meta


def extract_bgr_frame_from_netcdf(
    nc_path: str | Path,
    *,
    time_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    依次尝试变量组合：
    1) ADT + UGOS + VGOS（中尺度涡 NC）
    2) SLA/SSH + (UGOS|SSU) + (VGOS|SSV)
    3) SST + SSU + SSV（海域要素预测 NC，演示用）

    返回 (BGR uint8 H×W×3, meta)。
    """
    a0, u0, v0, meta = extract_triple_scalar_fields_from_netcdf(nc_path, time_index=time_index)
    del u0, v0  # Fair-B0 3ch：推理输入为 ADT×3，与 config/eddy_v6_b0_fair 训练一致
    bgr = fair_adt_triple_bgr_u8(a0)
    meta = {**meta, "inference_stack": "fair_adt_x3", "inference_input_channels": 3}
    return bgr, meta
