"""从海域要素预测日 NetCDF 拼接时间维并构建 (T,H,W,C) 场。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.netcdf_io import open_netcdf_dataset
from src.utils.config import load_yaml, project_root, resolve_path


def _pick_dataarray(ds: xr.Dataset, candidates: list[str]) -> xr.DataArray:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"以下变量均不存在: {candidates}，当前 data_vars={list(ds.data_vars)}")


def load_variable_map() -> dict[str, Any]:
    return load_yaml(project_root() / "config/variable_map.yaml")


def stack_hydro_fields(
    nc_paths: list[Path],
    internal_features: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    按日文件顺序拼接 time，返回 field shape (T, H, W, C)，以及元信息。
    """
    vm = load_variable_map()
    ch_map: dict[str, list[str]] = vm.get("channels", {})
    chunks: list[np.ndarray] = []
    meta: dict[str, Any] = {"files_used": 0, "time_size_per_file": []}

    for fp in nc_paths:
        ds, tmp_copy = open_netcdf_dataset(fp)
        try:
            arrs = []
            for feat in internal_features:
                cands = ch_map.get(feat, [feat])
                da = _pick_dataarray(ds, cands)
                vals = da.values.astype(np.float32)
                arrs.append(vals)
            # (T, lat, lon) per channel -> stack C
            vol = np.stack(arrs, axis=-1)
            if vol.ndim != 4:
                raise ValueError(f"期望 4D (T,lat,lon,C)，得到 {vol.shape} @ {fp}")
            chunks.append(vol)
            meta["time_size_per_file"].append(vol.shape[0])
            meta["files_used"] += 1
        finally:
            ds.close()
            if tmp_copy is not None:
                try:
                    tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
                except OSError:
                    pass

    if not chunks:
        raise RuntimeError("未读取到任何 NetCDF 数据")

    field = np.concatenate(chunks, axis=0)
    meta["T"], meta["H"], meta["W"], meta["C"] = field.shape
    bad = int(np.sum(~np.isfinite(field)))
    if bad:
        print(
            f"警告: 拼接场含 {bad} 个非有限值(nan/inf)，已用 0 替换（常见于 NetCDF 缺测/填充）。",
            flush=True,
        )
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        field = field.astype(np.float32)
    return field, meta


def _calendar_year_from_path(p: Path) -> int | None:
    """从路径中解析年份目录名（如 .../1994/19940101.nc → 1994）。"""
    for part in p.parts:
        if part.isdigit() and len(part) == 4:
            y = int(part)
            if 1900 <= y <= 2100:
                return y
    return None


def discover_hydro_nc_paths(
    raw_root: Path,
    hydro_subdir: str,
    max_daily_files: int | None = None,
    *,
    years: set[int] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[Path]:
    """按文件名 YYYYMMDD.nc 排序；可选按父目录年份过滤（命题方 train/val/test 年）。"""
    sub = raw_root / hydro_subdir
    if not sub.is_dir():
        raise FileNotFoundError(f"水文数据目录不存在: {sub}")
    files = sorted(sub.rglob("*.nc"))
    files = [f for f in files if "__MACOSX" not in f.parts]
    if not files:
        raise FileNotFoundError(f"未在 {sub} 下发现 .nc 文件")

    if years is not None:
        files = [f for f in files if _calendar_year_from_path(f) in years]
    elif year_min is not None and year_max is not None:
        files = [
            f
            for f in files
            if (y := _calendar_year_from_path(f)) is not None and year_min <= y <= year_max
        ]

    def sort_key(p: Path) -> tuple:
        stem = p.stem
        if stem.isdigit() and len(stem) >= 8:
            return (0, stem)
        return (1, str(p))

    files = sorted(files, key=sort_key)
    if max_daily_files is not None and max_daily_files > 0:
        files = files[: max_daily_files]
    return files


def build_windows(
    field: np.ndarray,
    input_steps: int,
    output_steps: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    field: (T, H, W, C)
    返回 X (N, T_in, H, W, C), y (N, T_out, H, W, C)
    """
    t, h, w, c = field.shape
    need = input_steps + output_steps
    if t < need:
        raise ValueError(
            f"时间长度 T={t} 不足以覆盖 input_steps+output_steps={need}。"
            f"请增加 --max-daily-files（或合并更多日 NetCDF），或改用更小的 input_steps/output_steps。"
        )
    starts = list(range(0, t - need + 1, stride))
    n = len(starts)
    item = np.dtype(np.float32).itemsize
    bytes_xy = n * (input_steps + output_steps) * h * w * c * item
    gib = bytes_xy / (1024**3)
    print(
        f"滑窗: N={n}, stride={stride}, 预计数组 X+y 约 {gib:.2f} GiB（float32）；"
        f"若过大将极慢或似卡死，请加大 --stride 或减少日文件数。",
        flush=True,
    )
    if gib > 8:
        print(
            "警告: 建议 --stride 12～24（或更大）以减小 N；全量训练可后续用 Dataset 按需读 nc。",
            file=sys.stderr,
            flush=True,
        )
    x = np.empty((n, input_steps, h, w, c), dtype=np.float32)
    y = np.empty((n, output_steps, h, w, c), dtype=np.float32)
    report_every = max(1, min(500, n // 20))
    for i, s in enumerate(starts):
        if i % report_every == 0 or i == n - 1:
            print(f"  滑窗填充 {i + 1}/{n}", flush=True)
        x[i] = field[s : s + input_steps]
        y[i] = field[s + input_steps : s + input_steps + output_steps]
    return x, y


def split_train_val_test(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """按样本顺序切段（时间顺序的前 80%/10%/10% 窗口）。"""
    n = x.shape[0]
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return (
        (x[:i1], y[:i1]),
        (x[i1:i2], y[i1:i2]),
        (x[i2:], y[i2:]),
    )


def zscore_fit(
    x_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对 (N,T,H,W,C) 在 N,T,H,W 上求每通道 mean/std（忽略 nan，避免 mean/std 为 nan）。"""
    x = np.asarray(x_train, dtype=np.float64)
    if not np.isfinite(x).all():
        n_bad = int(np.size(x) - np.sum(np.isfinite(x)))
        print(
            f"警告: 训练张量仍含 {n_bad} 个非有限值，Z-score 前已清理。",
            flush=True,
        )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.nanmean(x, axis=(0, 1, 2, 3), keepdims=True)
    std = np.nanstd(x, axis=(0, 1, 2, 3), keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (x.astype(np.float64) - mean) / std
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)
