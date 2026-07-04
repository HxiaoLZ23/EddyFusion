"""xarray 懒加载 + 时空 ROI 裁剪 + 变量映射探测（论文 §5.2）。"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from src.preprocess.netcdf_io import open_netcdf_dataset, write_xarray_to_netcdf_via_temp
from src.utils.config import load_yaml, project_root

_SPATIAL_DIMS = frozenset(
    {"latitude", "longitude", "lat", "lon", "Latitude", "Longitude", "x", "y", "X", "Y"}
)

NC_SUBSET_DIR = project_root() / "app" / "data" / "nc_uploads" / "subsets"


def load_nc_variable_map() -> dict[str, Any]:
    return load_yaml("config/nc_variable_map.yaml")


def _lower_var_index(ds: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in list(getattr(ds, "data_vars", [])):
        out[str(k).lower()] = str(k)
    for k in list(getattr(ds, "coords", [])):
        lk = str(k).lower()
        if lk not in out:
            out[lk] = str(k)
    return out


def _pick_name(ds: Any, aliases: list[str]) -> str | None:
    idx = _lower_var_index(ds)
    for a in aliases:
        hit = idx.get(str(a).lower())
        if hit is not None:
            return hit
    return None


def detect_coord_names(ds: Any) -> dict[str, str | None]:
    m = load_nc_variable_map()
    coords = m.get("coordinates") or {}
    return {
        "time": _pick_name(ds, list(coords.get("time") or ["time"])),
        "lat": _pick_name(ds, list(coords.get("lat") or ["lat", "latitude"])),
        "lon": _pick_name(ds, list(coords.get("lon") or ["lon", "longitude"])),
    }


def probe_nc_meta(nc_path: str | Path) -> dict[str, Any]:
    """打开一次 NC，返回时间/空间范围与变量映射摘要（不物化全量数组）。"""
    path = Path(nc_path).expanduser().resolve()
    ds, tmp = open_netcdf_dataset(path)
    try:
        names = detect_coord_names(ds)
        tname, latn, lonn = names["time"], names["lat"], names["lon"]
        meta: dict[str, Any] = {
            "source_path": str(path),
            "variables": sorted(str(k) for k in ds.data_vars),
            "dimensions": {str(k): int(ds.sizes[k]) for k in ds.dims},
            "coords": names,
            "variable_map": build_variable_map_report(ds),
        }
        if tname and tname in ds.dims:
            meta["time_len"] = int(ds.sizes[tname])
            try:
                tvals = ds.coords[tname].values
                if len(tvals):
                    meta["time_start_label"] = str(tvals[0])
                    meta["time_end_label"] = str(tvals[-1])
            except Exception:
                pass
        if latn and latn in ds.coords:
            latv = ds.coords[latn].values
            if len(latv):
                meta["lat_min"] = float(min(latv))
                meta["lat_max"] = float(max(latv))
        if lonn and lonn in ds.coords:
            lonv = ds.coords[lonn].values
            if len(lonv):
                meta["lon_min"] = float(min(lonv))
                meta["lon_max"] = float(max(lonv))
        return meta
    finally:
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def build_variable_map_report(ds: Any) -> dict[str, Any]:
    m = load_nc_variable_map()
    vars_cfg: dict[str, Any] = m.get("variables") or {}
    found: dict[str, str] = {}
    missing: list[str] = []
    for std, cfg in vars_cfg.items():
        if std in ("time", "lat", "lon"):
            continue
        aliases = list((cfg or {}).get("aliases") or [])
        hit = _pick_name(ds, aliases)
        if hit:
            found[std] = hit
        else:
            missing.append(std)
    eddy_ok = (found.get("adt") or found.get("sla")) and found.get("u") and found.get("v")
    wind_ok = (found.get("u10") or found.get("v10")) and found.get("swh")
    return {
        "found": found,
        "missing_standard": missing,
        "eddy_ready": bool(eddy_ok),
        "windwave_ready": bool(wind_ok),
    }


def _time_dim_on_da(ds: Any, tname: str | None) -> str | None:
    if tname and tname in ds.dims:
        return tname
    for d in ds.dims:
        if d not in _SPATIAL_DIMS and str(d).lower() not in ("x", "y"):
            return str(d)
    return None


def subset_netcdf(
    nc_path: str | Path,
    *,
    time_start: int | None = None,
    time_stop: int | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """
    懒加载打开 → sel 时间/空间 → 写出子集 NC。
    返回仓库相对路径与 meta（供 React 左栏 TaskConfigPanel）。
    """
    path = Path(nc_path).expanduser().resolve()
    ds, tmp = open_netcdf_dataset(path)
    try:
        names = detect_coord_names(ds)
        tname = _time_dim_on_da(ds, names["time"])
        latn, lonn = names["lat"], names["lon"]
        subset = ds

        applied: dict[str, Any] = {"time": None, "bbox": None}

        if tname is not None:
            tlen = int(subset.sizes[tname])
            i0 = 0 if time_start is None else max(0, min(int(time_start), tlen - 1))
            i1 = tlen - 1 if time_stop is None else max(0, min(int(time_stop), tlen - 1))
            if i0 > i1:
                i0, i1 = i1, i0
            if i0 != 0 or i1 != tlen - 1:
                subset = subset.isel({tname: slice(i0, i1 + 1)})
                applied["time"] = {"start_index": i0, "stop_index": i1, "dim": tname}

        if latn and lonn and latn in subset.dims and lonn in subset.dims:
            sel_kw: dict[str, slice] = {}
            if lat_min is not None or lat_max is not None:
                lo = lat_min if lat_min is not None else float(subset[latn].min())
                hi = lat_max if lat_max is not None else float(subset[latn].max())
                if lo > hi:
                    lo, hi = hi, lo
                sel_kw[latn] = slice(lo, hi)
            if lon_min is not None or lon_max is not None:
                lo = lon_min if lon_min is not None else float(subset[lonn].min())
                hi = lon_max if lon_max is not None else float(subset[lonn].max())
                if lo > hi:
                    lo, hi = hi, lo
                sel_kw[lonn] = slice(lo, hi)
            if sel_kw:
                subset = subset.sel(sel_kw)
                applied["bbox"] = {
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lat_dim": latn,
                    "lon_dim": lonn,
                }

        map_report = build_variable_map_report(subset)
        if task == "eddy" and not map_report.get("eddy_ready"):
            raise ValueError("所选子集缺少涡旋所需变量（ADT/SLA + U + V）")
        if task == "windwave" and not map_report.get("windwave_ready"):
            raise ValueError("所选子集缺少风浪所需变量（风场 + SWH）")

        NC_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:10]
        digest = hashlib.sha1(str(path).encode() + repr(applied).encode()).hexdigest()[:8]
        out_name = f"subset_{tag}_{digest}.nc"
        out_abs = (NC_SUBSET_DIR / out_name).resolve()
        write_xarray_to_netcdf_via_temp(subset, out_abs)

        rel = out_abs.relative_to(project_root().resolve()).as_posix()
        return {
            "status": "ok",
            "nc_path": rel,
            "source_nc_path": str(path),
            "task": task,
            "applied": applied,
            "dimensions": {str(k): int(subset.sizes[k]) for k in subset.dims},
            "variable_map": map_report,
            "size_mb": round(out_abs.stat().st_size / (1024**2), 4),
        }
    finally:
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
