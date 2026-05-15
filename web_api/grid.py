from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.hydro_nc_infer_build import sort_hydro_nc_paths
from src.preprocess.hydro_nc_stack import load_variable_map
from src.preprocess.netcdf_io import open_netcdf_dataset
from src.utils.config import load_yaml


def _pick_dataarray(ds: Any, candidates: list[str]) -> Any:
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(f"以下变量均不存在: {candidates}，当前 data_vars={list(ds.data_vars)}")


def lonlat_from_hydro_nc(nc_path: Path, *, config_path: str) -> tuple[list[float], list[float]]:
    """
    从首份 NC 的要素变量上读取 1D lat / lon（与 stack_hydro_fields 维度顺序一致）。
    """
    cfg = load_yaml(config_path)
    feats = list(cfg["data"]["input_features"])
    feat0 = feats[0]
    vm = load_variable_map()
    cands = vm.get("channels", {}).get(feat0, [feat0])

    ds, tmp = open_netcdf_dataset(nc_path)
    try:
        da = _pick_dataarray(ds, cands)
        dims = list(da.dims)
        lat_names = ("lat", "LAT", "latitude", "Latitude", "nav_lat", "ylat")
        lon_names = ("lon", "LON", "longitude", "Longitude", "nav_lon", "xlon")
        lat_dim = next((d for d in dims if str(d).lower() in {n.lower() for n in lat_names}), None)
        lon_dim = next((d for d in dims if str(d).lower() in {n.lower() for n in lon_names}), None)
        if lat_dim is None or lon_dim is None:
            raise ValueError(f"无法从维度 {dims} 推断 lat/lon")

        lat_coord = da.coords[lat_dim]
        lon_coord = da.coords[lon_dim]
        lats = np.asarray(lat_coord.values, dtype=float)
        lons = np.asarray(lon_coord.values, dtype=float)
        if lats.ndim == 2:
            lats = lats[:, 0]
        if lons.ndim == 2:
            lons = lons[0, :]
        if lats.ndim != 1 or lons.ndim != 1:
            raise ValueError(f"lat/lon 需为 1D，得到 lat.shape={lats.shape}, lon.shape={lons.shape}")
        return [float(x) for x in lons], [float(x) for x in lats]
    finally:
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


def sorted_nc_paths(paths: list[Path]) -> list[Path]:
    return sort_hydro_nc_paths([p.resolve() for p in paths])
