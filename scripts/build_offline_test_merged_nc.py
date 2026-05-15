#!/usr/bin/env python3
"""
从「服创数据集」中尺度涡 / 海域要素预测 / 风浪异常识别 各取一段，
合并为 **单份 NetCDF**（时间步数由 ``--n-time`` 指定，默认 20），供离线系统或 Facade 烟测。

设计说明
--------
- **水文 + 风浪（风、浪）**：以海域要素预测 **按日文件顺序拼接** 后的 ``time`` 为主轴，
  从 ``--hydro-day`` 所指文件起向后拼接，直至 ``time`` 长度 ≥ ``--n-time``，再截取前 ``n_time`` 步。
  风浪 ``oper`` / ``wave`` 按主时间轴的 **年月** 自动合并多月 NetCDF 后，将 ``u10/v10/swh``
  **线性插值**到水文 ``time, lat, lon``（需 **scipy**，见根目录 ``requirements.txt``）。
- **涡旋**：``adt/ugos/vgos`` 置于 ``time_eddy × latitude_eddy × longitude_eddy``（连续 ``n_time`` 日），
  与主 ``time`` 日历可不同步，仅作同文件多源烟测。

用法（仓库根）::

    python scripts/build_offline_test_merged_nc.py
    python scripts/build_offline_test_merged_nc.py --n-time 300 --out data/processed/demo/offline_test_merged_t300.nc

依赖：xarray、netCDF4、scipy。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.preprocess.netcdf_io import open_netcdf_dataset, write_xarray_to_netcdf_via_temp
from src.utils.config import project_root, resolve_path


def _discover_subdirs(raw: Path) -> tuple[Path, Path, Path]:
    """按 rglob *.nc 数量识别：涡旋(最少)、风浪(中等)、水文(最多)。"""
    pairs: list[tuple[Path, int]] = []
    for d in raw.iterdir():
        if d.is_dir():
            pairs.append((d, len(list(d.rglob("*.nc")))))
    if len(pairs) < 3:
        raise FileNotFoundError(f"服创数据集下子目录不足 3 个: {raw}")
    pairs.sort(key=lambda x: x[1])
    eddy_dir, ww_dir, hydro_dir = pairs[0][0], pairs[1][0], pairs[2][0]
    return eddy_dir, ww_dir, hydro_dir


def _open(p: Path) -> tuple[xr.Dataset, Path | None]:
    ds, tmp = open_netcdf_dataset(p)
    return ds, tmp


def _close(ds: xr.Dataset, tmp: Path | None) -> None:
    ds.close()
    if tmp is not None:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _load_hydro_concat(
    hydro_dir: Path,
    year: str,
    start_nc_name: str,
    n_time: int,
) -> tuple[xr.Dataset, list[Path]]:
    """从 ``year/start_nc_name`` 起按文件名顺序拼接日文件，直至 time ≥ n_time。"""
    ydir = hydro_dir / year
    if not ydir.is_dir():
        raise FileNotFoundError(f"水文年目录不存在: {ydir}")
    files = sorted(ydir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"水文目录无 nc: {ydir}")
    start_idx = next((i for i, f in enumerate(files) if f.name == start_nc_name), None)
    if start_idx is None:
        raise FileNotFoundError(f"未找到起始水文文件 {start_nc_name} 于 {ydir}")
    parts: list[xr.Dataset] = []
    used: list[Path] = []
    total = 0
    for fp in files[start_idx:]:
        ds, tmp = _open(fp)
        try:
            sub = ds.load()
        finally:
            _close(ds, tmp)
        parts.append(sub)
        used.append(fp)
        total += int(sub.sizes["time"])
        if total >= n_time:
            break
    if total < n_time:
        raise ValueError(f"水文拼接后 time 总长 {total} < 需要 {n_time}，请换起始日或换有数据的年。")
    hcat = xr.concat(parts, dim="time")
    hcat = hcat.sortby("time")
    # 去重 time（相邻日文件边界可能重复）
    _, uniq = np.unique(hcat["time"].values, return_index=True)
    hcat = hcat.isel(time=uniq)
    if int(hcat.sizes["time"]) < n_time:
        raise ValueError(f"去重后 time 长度 {hcat.sizes['time']} < {n_time}")
    hout = hcat.isel(time=slice(0, n_time))
    return hout, used


def _ym_keys_from_times(times: np.ndarray) -> list[str]:
    """返回 'YYYYMM' 列表（按时间顺序去重）。"""
    ts = pd.to_datetime(times)
    keys: list[str] = []
    seen: set[str] = set()
    for t in ts:
        k = f"{t.year}{t.month:02d}"
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _concat_ww_month(
    ww_dir: Path,
    year: str,
    ym_keys: list[str],
    *,
    kind: str,
) -> tuple[xr.Dataset, list[Path]]:
    """kind: 'oper' 或 'wave'，合并多个月份目录下的首个匹配 nc。"""
    paths: list[Path] = []
    chunks: list[xr.Dataset] = []
    for ym in ym_keys:
        sub = ww_dir / year / ym
        if not sub.is_dir():
            raise FileNotFoundError(f"风浪子目录不存在: {sub}")
        pat = "*oper*.nc" if kind == "oper" else "*wave*.nc"
        cand = sorted(sub.glob(pat))
        if not cand:
            raise FileNotFoundError(f"{sub} 下无 {pat}")
        fp = cand[0]
        paths.append(fp)
        ds, tmp = _open(fp)
        try:
            chunks.append(ds.load())
        finally:
            _close(ds, tmp)
    if len(chunks) == 1:
        return _prep_ww_time_dim(chunks[0]), paths
    preped = [_prep_ww_time_dim(c) for c in chunks]
    return xr.concat(preped, dim="time"), paths


def _prep_ww_time_dim(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        return ds.rename({"valid_time": "time"})
    return ds


def build_merged(
    *,
    raw_root: Path,
    hydro_year: str,
    hydro_day_file: str,
    ww_year: str,
    ww_month: str | None,
    n_time: int,
    eddy_time_start: int,
) -> tuple[xr.Dataset, dict[str, Any]]:
    eddy_dir, ww_dir, hydro_dir = _discover_subdirs(raw_root)
    meta: dict[str, Any] = {
        "eddy_dir": str(eddy_dir),
        "hydro_dir": str(hydro_dir),
        "windwave_dir": str(ww_dir),
    }

    h20, hydro_paths = _load_hydro_concat(hydro_dir, hydro_year, hydro_day_file, n_time)

    ym_keys = _ym_keys_from_times(h20["time"].values)
    if ww_month is not None and str(ww_month) not in ym_keys:
        ym_keys = ym_keys + [str(ww_month)]

    ds_o_full, oper_paths = _concat_ww_month(ww_dir, ww_year, ym_keys, kind="oper")
    ds_w_full, wave_paths = _concat_ww_month(ww_dir, ww_year, ym_keys, kind="wave")
    ds_o_full = ds_o_full.sortby("time")
    ds_w_full = ds_w_full.sortby("time")
    _, uo = np.unique(ds_o_full["time"].values, return_index=True)
    _, uw = np.unique(ds_w_full["time"].values, return_index=True)
    ds_o_full = ds_o_full.isel(time=uo)
    ds_w_full = ds_w_full.isel(time=uw)

    u10 = ds_o_full["u10"].rename({"latitude": "lat", "longitude": "lon"})
    v10 = ds_o_full["v10"].rename({"latitude": "lat", "longitude": "lon"})
    swh = ds_w_full["swh"].rename({"latitude": "lat", "longitude": "lon"})

    u10_i = u10.interp(time=h20["time"], lat=h20["lat"], lon=h20["lon"], method="linear").load()
    v10_i = v10.interp(time=h20["time"], lat=h20["lat"], lon=h20["lon"], method="linear").load()
    swh_i = swh.interp(time=h20["time"], lat=h20["lat"], lon=h20["lon"], method="linear").load()

    eddy_nc = sorted(eddy_dir.glob("*.nc"))[0]
    ds_e, tmp_e = _open(eddy_nc)
    try:
        ne = int(ds_e.sizes["time"])
        if ne < n_time:
            raise ValueError(f"涡旋 time 长度 {ne} < 需要 {n_time}")
        e0 = max(0, min(int(eddy_time_start), ne - n_time))
        ed = ds_e.isel(time=slice(e0, e0 + n_time)).load()
        ed = ed.rename(
            {
                "time": "time_eddy",
                "latitude": "latitude_eddy",
                "longitude": "longitude_eddy",
            }
        )
    finally:
        _close(ds_e, tmp_e)

    merged = xr.merge(
        [
            h20,
            xr.Dataset({"u10": u10_i, "v10": v10_i, "swh": swh_i}),
            ed[["adt", "ugos", "vgos"]],
        ],
        compat="override",
    )
    merged.attrs["title"] = f"offline_test_merged_t{n_time}"
    merged.attrs["merge_note"] = (
        f"主网格 time×lat×lon（time={n_time}）：水文 sst/sss/ssu/ssv + 风浪 u10/v10/swh（插值到水文格点）；"
        "涡旋 adt/ugos/vgos 在 time_eddy×latitude_eddy×longitude_eddy，与主 time 日历可不同步，仅作同文件多源烟测。"
    )
    merged.attrs["sources"] = json.dumps(
        {
            "hydro_paths": [str(p) for p in hydro_paths],
            "wind_oper_paths": [str(p) for p in oper_paths],
            "wind_wave_paths": [str(p) for p in wave_paths],
            "eddy": str(eddy_nc),
            "eddy_time_slice": [e0, e0 + n_time],
            "n_time": n_time,
        },
        ensure_ascii=False,
    )
    meta["output_dims"] = {str(k): int(v) for k, v in merged.sizes.items()}
    meta["hydro_paths"] = [str(p) for p in hydro_paths]
    meta["wind_oper_paths"] = [str(p) for p in oper_paths]
    meta["wind_wave_paths"] = [str(p) for p in wave_paths]
    meta["eddy_path"] = str(eddy_nc)
    meta["eddy_time_start"] = e0
    return merged, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=str,
        default="data/processed/demo/offline_test_merged_t20.nc",
        help="输出 NetCDF（相对仓库根）",
    )
    ap.add_argument("--hydro-year", default="2015")
    ap.add_argument("--hydro-day", default="20150102.nc")
    ap.add_argument("--ww-year", default="2015")
    ap.add_argument(
        "--ww-month",
        default=None,
        help="可选：强制并入该风浪月目录 YYYYMM（默认仅按水文时间轴自动选月）",
    )
    ap.add_argument("--n-time", type=int, default=20)
    ap.add_argument("--eddy-time-start", type=int, default=0, help="涡旋文件内起始日索引")
    args = ap.parse_args()

    root = project_root()
    raw = root / "服创数据集"
    if not raw.is_dir():
        raise SystemExit(f"未找到数据目录: {raw}")

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    merged, meta = build_merged(
        raw_root=raw,
        hydro_year=args.hydro_year,
        hydro_day_file=args.hydro_day,
        ww_year=args.ww_year,
        ww_month=args.ww_month,
        n_time=int(args.n_time),
        eddy_time_start=int(args.eddy_time_start),
    )

    encoding: dict[str, Any] = {}
    for v in merged.data_vars:
        encoding[v] = {"zlib": True, "complevel": 3}

    write_xarray_to_netcdf_via_temp(merged, out, encoding=encoding)
    meta_path = out.with_suffix(".source_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
