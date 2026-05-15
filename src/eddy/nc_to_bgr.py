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
    path = Path(nc_path).expanduser().resolve()
    ds, tmp_copy = open_netcdf_dataset(path)
    meta: dict[str, Any] = {"nc_path": str(path)}
    if tmp_copy is not None:
        meta["opened_via_tmp"] = str(tmp_copy)

    try:
        candidates: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = [
            (("adt", "ADT"), ("ugos", "UGOS"), ("vgos", "VGOS"), "adt_ugos_vgos"),
            (("sla", "SLA", "ssh", "SSH"), ("ugos", "UGOS", "ssu", "SSU"), ("vgos", "VGOS", "ssv", "SSV"), "sla_uv"),
            (("sst", "SST"), ("ssu", "SSU"), ("ssv", "SSV"), "sst_ssu_ssv"),
        ]
        last_err: Exception | None = None
        for n0, n1, n2, tag in candidates:
            try:
                ch0 = _pick_dataarray(ds, n0)
                ch1 = _pick_dataarray(ds, n1)
                ch2 = _pick_dataarray(ds, n2)
            except KeyError as e:
                last_err = e
                continue

            tname = _time_dim_name(ch0)
            if tname is not None:
                T = int(ch0.sizes[tname])
                ti = max(0, min(int(time_index), T - 1))
                a0 = _isel_time(ch0, tname, ti)
                u0 = _isel_time(ch1, tname, ti)
                v0 = _isel_time(ch2, tname, ti)
                meta["time_dim"] = tname
                meta["time_len"] = T
                meta["time_index"] = ti
                tl = _time_label_for_index(ch0, tname, ti)
                if tl:
                    meta["time_label"] = tl
            else:
                a0 = _spatial_only_vals(ch0)
                u0 = _spatial_only_vals(ch1)
                v0 = _spatial_only_vals(ch2)
                meta["time_dim"] = None

            if a0.shape != u0.shape or a0.shape != v0.shape:
                last_err = ValueError(f"三通道形状不一致 {a0.shape} {u0.shape} {v0.shape}")
                continue

            meta["channel_triple"] = tag
            return a0.astype(np.float64), u0.astype(np.float64), v0.astype(np.float64), meta

        raise ValueError(
            f"无法从 NC 匹配 ADT/流场或 SST/流场变量组合。最后错误: {last_err!r}；"
            f"data_vars={list(ds.data_vars)}"
        ) from last_err
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


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
    rgb = np.stack([_norm_to_u8(a0), _norm_to_u8(u0), _norm_to_u8(v0)], axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, meta
