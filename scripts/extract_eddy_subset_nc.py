#!/usr/bin/env python3
"""
从命题方「中尺度涡识别」大体积 NetCDF 截取较小时空窗口，便于涡旋页视频烟测（场更小、耗时更短）。

默认按 config/data.yaml 的 raw_root + eddy_subdir 查找 `--stem` 对应文件；
也可用 `--src` 指定任意路径。

示例（在仓库根执行）::

  python scripts/extract_eddy_subset_nc.py --info-only
  python scripts/extract_eddy_subset_nc.py --t-count 24 --spatial-size 320
  python scripts/extract_eddy_subset_nc.py --src 服创数据集/中尺度涡识别/19930101_20021231.nc --out outputs/eddy_subset_demo.nc
  python scripts/extract_eddy_subset_nc.py --out outputs/eddy_subset_19930101_20021231_small.nc --synthetic-windwave

写出时经 ``%TEMP%`` 英文临时文件再 ``shutil.copy2`` 到 ``--out``，避免 Windows 中文路径下 netCDF4 直接写入失败。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import xarray as xr

from src.preprocess.netcdf_io import open_netcdf_dataset, write_xarray_to_netcdf_via_temp
from src.utils.config import load_yaml, resolve_path


def _reference_field_for_grid(ds: xr.Dataset) -> xr.DataArray:
    for key in ("adt", "ADT", "sla", "SLA", "sst", "SST"):
        lk = {str(k).lower(): k for k in ds.data_vars}
        if key.lower() in lk:
            return ds[lk[key.lower()]]
    for _k, v in ds.data_vars.items():
        return v
    raise ValueError("数据集中无 data_vars")


def _attach_synthetic_wind_wave(sub_ds: xr.Dataset, *, seed: int) -> xr.Dataset:
    """与现有格点/时间维对齐，附加演示用 u10/v10/swh（供涡旋页路径① + 风浪联动烟测）。"""
    ref = _reference_field_for_grid(sub_ds)
    dims = tuple(ref.dims)
    shape = tuple(int(ref.sizes[d]) for d in dims)
    coords = {d: sub_ds.coords[d] for d in dims if d in sub_ds.coords}

    rng = np.random.default_rng(int(seed))
    u10 = rng.normal(5.5, 1.2, shape).astype(np.float32)
    v10 = rng.normal(0.5, 1.5, shape).astype(np.float32)
    spd = np.sqrt(np.maximum(u10, 0.0) ** 2 + v10**2)
    swh = (0.07 * spd + 0.25 + 0.12 * rng.random(shape).astype(np.float32)).astype(np.float32)

    out = sub_ds.copy()
    out["u10"] = xr.DataArray(u10, dims=dims, coords=coords)
    out["v10"] = xr.DataArray(v10, dims=dims, coords=coords)
    out["swh"] = xr.DataArray(swh, dims=dims, coords=coords)
    out["u10"].attrs.setdefault("long_name", "10m wind u")
    out["v10"].attrs.setdefault("long_name", "10m wind v")
    out["swh"].attrs.setdefault("long_name", "significant wave height (synthetic demo)")
    out.attrs.setdefault("wind_wave_note", "synthetic u10/v10/swh for demo; not observation")
    return out


def _find_time_and_spatial_dims(ds: xr.Dataset) -> tuple[str | None, list[str]]:
    dim_names = list(ds.sizes.keys())
    tname: str | None = None
    for d in dim_names:
        dl = d.lower()
        if "time" in dl or dl in ("t", "date"):
            tname = d
            break
    spatial: list[str] = []
    for d in dim_names:
        if tname is not None and d == tname:
            continue
        dl = d.lower()
        if "bnd" in dl or dl in ("nv", "nvertices"):
            continue
        spatial.append(d)
    return tname, spatial


def _center_slice(n: int, size: int | None) -> slice:
    if size is None or int(size) <= 0 or n <= int(size):
        return slice(None)
    sz = int(size)
    start = max(0, (n - sz) // 2)
    end = min(n, start + sz)
    return slice(start, end)


def main() -> int:
    ap = argparse.ArgumentParser(description="截取涡旋 NC 时空子集")
    ap.add_argument("--data-config", type=str, default="config/data.yaml")
    ap.add_argument(
        "--src",
        type=str,
        default="",
        help="源 NC；留空则使用 --stem 在 raw_root/eddy_subdir 下查找",
    )
    ap.add_argument("--stem", type=str, default="19930101_20021231")
    ap.add_argument("--out", type=str, default="outputs/eddy_subset_19930101_20021231_small.nc")
    ap.add_argument("--t-start", type=int, default=0)
    ap.add_argument("--t-count", type=int, default=32)
    ap.add_argument(
        "--spatial-size",
        type=int,
        default=320,
        help="对每一空间维居中裁剪到至多该长度；0 表示不裁剪",
    )
    ap.add_argument("--info-only", action="store_true")
    ap.add_argument(
        "--synthetic-windwave",
        action="store_true",
        help="在子集上附加同维度的演示用 u10/v10/swh，便于涡旋页路径① 与风浪页联调",
    )
    ap.add_argument("--wind-seed", type=int, default=123, help="合成风浪场随机种子")
    args = ap.parse_args()

    if args.src.strip():
        src = resolve_path(args.src.strip())
    else:
        cfg = load_yaml(args.data_config)
        raw_root = resolve_path(cfg["paths"]["raw_root"])
        sub = cfg["paths"].get("eddy_subdir", "中尺度涡识别")
        src = raw_root / sub / f"{args.stem}.nc"

    if not src.is_file():
        print(f"找不到文件: {src}", file=sys.stderr)
        print(
            "请将命题方数据放到 服创数据集/中尺度涡识别/ 下，或使用 --src 指定完整相对路径。",
            file=sys.stderr,
        )
        return 1

    ds, tmp = open_netcdf_dataset(src)
    try:
        if args.info_only:
            print(ds)
            return 0

        tname, spatial = _find_time_and_spatial_dims(ds)
        sel: dict[str, slice] = {}

        if tname is not None:
            T = int(ds.sizes[tname])
            t0 = max(0, min(int(args.t_start), max(0, T - 1)))
            tc = max(1, int(args.t_count))
            t1 = min(T, t0 + tc)
            sel[tname] = slice(t0, t1)

        ssize = int(args.spatial_size)
        if ssize > 0:
            for dim in spatial:
                n = int(ds.sizes[dim])
                sel[dim] = _center_slice(n, ssize)

        sub_ds = ds.isel(**sel)
        if args.synthetic_windwave:
            sub_ds = _attach_synthetic_wind_wave(sub_ds, seed=args.wind_seed)

        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.resolve() == Path(src).resolve():
            print("输出不能与源文件相同", file=sys.stderr)
            return 1

        enc = {v: {"zlib": True, "complevel": 4} for v in sub_ds.data_vars}
        write_xarray_to_netcdf_via_temp(sub_ds, out_path, encoding=enc)

        try:
            rel = out_path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            rel = out_path
        print(f"已写入 {rel}")
        print(f"维度: {dict(sub_ds.sizes)}")
        if args.synthetic_windwave:
            print("已附加合成变量: u10, v10, swh（演示用，非实况）")
        return 0
    finally:
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
