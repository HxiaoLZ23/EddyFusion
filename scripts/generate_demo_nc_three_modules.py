#!/usr/bin/env python3
"""
生成三模块演示/测试用 NetCDF（与 variable_map、各模块入口变量约定一致）。

输出: outputs/demo_nc_three_modules/（默认在 .gitignore 下，本地生成即可）
  - mod1_ocean_sst_uv.nc   → 涡旋流场（SST+SSU+SSV）
  - mod2_ocean_wind_wave.nc → 风浪（u10,v10,swh）
  - mod3_ocean_hydro_L2.nc → 水文 L2 单文件 T=240
  - mod_fused_stream_windwave_video.nc → 流场+风浪同格同时间维（便于逐帧合成视频与路径①联调）

依赖: numpy, netCDF4（见 requirements.txt）

用法: python scripts/generate_demo_nc_three_modules.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 当以 `python scripts/本文件.py` 启动时，部分 Windows 环境 cwd 在 scripts/，会导致写 NC 异常；统一到仓库根
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from netCDF4 import Dataset

OUT = ROOT / "outputs" / "demo_nc_three_modules"


def _nc_path_for_write(path: Path) -> str:
    """Windows 下仓库路径含中文时，netCDF4 对绝对路径常报 PermissionError；chdir(ROOT) 后用相对路径写入。"""
    outp = path.resolve()
    try:
        rel = outp.relative_to(ROOT.resolve())
    except ValueError:
        rel = outp
    return str(rel).replace("\\", "/")


def _write_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _create_dims_and_coords(
    nc: Dataset,
    *,
    time_len: int,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> None:
    nc.createDimension("time", time_len)
    nc.createDimension("lat", len(lat_vals))
    nc.createDimension("lon", len(lon_vals))
    t = nc.createVariable("time", "i4", ("time",))
    t[:] = np.arange(time_len, dtype=np.int32)
    la = nc.createVariable("lat", "f4", ("lat",))
    la[:] = lat_vals.astype(np.float32)
    lo = nc.createVariable("lon", "f4", ("lon",))
    lo[:] = lon_vals.astype(np.float32)
    la.setncatts({"units": "degrees_north"})
    lo.setncatts({"units": "degrees_east"})


def make_eddy(out: Path) -> dict:
    rng = np.random.default_rng(42)
    nt, ny, nx = 2, 64, 64
    lat = np.linspace(31.0, 41.0, ny, dtype=np.float32)
    lon = np.linspace(117.0, 127.0, nx, dtype=np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    sst = np.empty((nt, ny, nx), dtype=np.float32)
    ssu = np.empty((nt, ny, nx), dtype=np.float32)
    ssv = np.empty((nt, ny, nx), dtype=np.float32)
    for it in range(nt):
        base = 22.0 + 0.4 * np.sin(xx * 0.15) * np.cos(yy * 0.12) + 0.15 * it
        sst[it] = base + rng.standard_normal((ny, nx)).astype(np.float32) * 0.3
        ssu[it] = 0.05 * np.sin(yy * 0.1 + it) + rng.standard_normal((ny, nx)).astype(np.float32) * 0.02
        ssv[it] = 0.05 * np.cos(xx * 0.1 - it) + rng.standard_normal((ny, nx)).astype(np.float32) * 0.02
    out.parent.mkdir(parents=True, exist_ok=True)
    outp = out.resolve()
    if outp.exists():
        outp.unlink(missing_ok=True)
    # Windows 下部分环境对 Path 对象打开 NC 会 PermissionError，显式 str
    nc = Dataset(_nc_path_for_write(outp), "w", format="NETCDF4")
    try:
        nc.setncatts({"title": "mod1 SST+SSU+SSV for vortex RGB pipeline", "module_hint": "vortex"})
        _create_dims_and_coords(nc, time_len=nt, lat_vals=lat, lon_vals=lon)
        for name, arr in (("SST", sst), ("SSU", ssu), ("SSV", ssv)):
            v = nc.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
            v[:] = arr
    finally:
        nc.close()
    return {"file": str(out.relative_to(ROOT)), "shape": {"time": nt, "lat": ny, "lon": nx}, "vars": ["SST", "SSU", "SSV"]}


def make_windwave(out: Path) -> dict:
    rng = np.random.default_rng(43)
    nt, ny, nx = 96, 12, 12
    lat = np.linspace(30.0, 40.0, ny, dtype=np.float32)
    lon = np.linspace(118.0, 126.0, nx, dtype=np.float32)
    u10 = rng.normal(6.0, 1.5, (nt, ny, nx)).astype(np.float32)
    v10 = rng.normal(0.0, 2.0, (nt, ny, nx)).astype(np.float32)
    spd = np.sqrt(np.maximum(u10, 0) ** 2 + v10**2)
    swh = (0.08 * spd + 0.3 * rng.random((nt, ny, nx)).astype(np.float32)).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    outp = out.resolve()
    if outp.exists():
        outp.unlink(missing_ok=True)
    # Windows 下部分环境对 Path 对象打开 NC 会 PermissionError，显式 str
    nc = Dataset(_nc_path_for_write(outp), "w", format="NETCDF4")
    try:
        nc.setncatts({"title": "mod2 u10 v10 swh", "module_hint": "wind_wave"})
        _create_dims_and_coords(nc, time_len=nt, lat_vals=lat, lon_vals=lon)
        for name, arr in (("u10", u10), ("v10", v10), ("swh", swh)):
            v = nc.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
            v[:] = arr
    finally:
        nc.close()
    return {"file": str(out.relative_to(ROOT)), "shape": {"time": nt, "lat": ny, "lon": nx}, "vars": ["u10", "v10", "swh"]}


def make_hydro_l2(out: Path) -> dict:
    rng = np.random.default_rng(44)
    nt, ny, nx = 240, 138, 125
    lat = np.linspace(31.0, 41.0, ny, dtype=np.float32)
    lon = np.linspace(117.0, 127.0, nx, dtype=np.float32)
    t1 = np.linspace(0, 2 * np.pi, nt, dtype=np.float32)[:, None, None]
    yy = np.linspace(0, 1, ny, dtype=np.float32)[None, :, None]
    xx = np.linspace(0, 1, nx, dtype=np.float32)[None, None, :]
    sst = (20.0 + 0.8 * np.sin(t1) + 0.3 * np.sin(4 * xx) * np.cos(3 * yy)).astype(np.float32)
    sst = sst + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.05
    sss = (34.5 + 0.1 * np.cos(t1 * 0.5) + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.02).astype(np.float32)
    ssu = (0.15 * np.sin(t1 + yy) + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.02).astype(np.float32)
    ssv = (0.15 * np.cos(t1 - xx) + rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.02).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    outp = out.resolve()
    if outp.exists():
        outp.unlink(missing_ok=True)
    # Windows 下部分环境对 Path 对象打开 NC 会 PermissionError，显式 str
    nc = Dataset(_nc_path_for_write(outp), "w", format="NETCDF4")
    try:
        nc.setncatts(
            {
                "title": "mod3 hydro TSUV L2 window T=240",
                "module_hint": "hydro",
            }
        )
        _create_dims_and_coords(nc, time_len=nt, lat_vals=lat, lon_vals=lon)
        pairs = (("SST", sst), ("sss", sss), ("SSU", ssu), ("SSV", ssv))
        for name, arr in pairs:
            v = nc.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
            v[:] = arr
    finally:
        nc.close()
    return {
        "file": str(out.relative_to(ROOT)),
        "shape": {"time": nt, "lat": ny, "lon": nx},
        "vars": ["SST", "sss", "SSU", "SSV"],
        "T_equals_input_plus_output": nt,
    }


def make_fused_stream_windwave_video(out: Path) -> dict:
    """流场（SST/SSU/SSV）与风浪（u10/v10/swh）同 (time,lat,lon)，适合抽多帧做视频与涡旋页路径①。"""
    rng = np.random.default_rng(45)
    nt, ny, nx = 48, 64, 64
    lat = np.linspace(31.0, 41.0, ny, dtype=np.float32)
    lon = np.linspace(117.0, 127.0, nx, dtype=np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    sst = np.empty((nt, ny, nx), dtype=np.float32)
    ssu = np.empty((nt, ny, nx), dtype=np.float32)
    ssv = np.empty((nt, ny, nx), dtype=np.float32)
    u10 = np.empty((nt, ny, nx), dtype=np.float32)
    v10 = np.empty((nt, ny, nx), dtype=np.float32)
    swh = np.empty((nt, ny, nx), dtype=np.float32)
    for it in range(nt):
        ph = 0.25 * float(it)
        base = 22.0 + 0.45 * np.sin(xx * 0.14 + ph) * np.cos(yy * 0.11 - 0.5 * ph)
        sst[it] = base + rng.standard_normal((ny, nx)).astype(np.float32) * 0.25
        ssu[it] = 0.08 * np.sin(yy * 0.12 + ph) + rng.standard_normal((ny, nx)).astype(np.float32) * 0.02
        ssv[it] = 0.08 * np.cos(xx * 0.13 - ph) + rng.standard_normal((ny, nx)).astype(np.float32) * 0.02
        u = 6.0 + 1.8 * np.sin(ph * 0.7) + 0.4 * np.sin(xx * 0.05 + ph)
        v = 1.2 * np.cos(ph * 0.5) + 0.5 * np.cos(yy * 0.06)
        u10[it] = (u + rng.standard_normal((ny, nx)).astype(np.float32) * 0.35).astype(np.float32)
        v10[it] = (v + rng.standard_normal((ny, nx)).astype(np.float32) * 0.35).astype(np.float32)
        spd = np.sqrt(np.maximum(u10[it], 0.0) ** 2 + v10[it] ** 2)
        swh[it] = (0.07 * spd + 0.25 + 0.15 * rng.random((ny, nx)).astype(np.float32)).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    outp = out.resolve()
    if outp.exists():
        outp.unlink(missing_ok=True)
    nc = Dataset(_nc_path_for_write(outp), "w", format="NETCDF4")
    try:
        nc.setncatts(
            {
                "title": "fused SST/UV + wind/wave for video demo",
                "module_hint": "vortex+wind_wave",
            }
        )
        _create_dims_and_coords(nc, time_len=nt, lat_vals=lat, lon_vals=lon)
        for name, arr in (
            ("SST", sst),
            ("SSU", ssu),
            ("SSV", ssv),
            ("u10", u10),
            ("v10", v10),
            ("swh", swh),
        ):
            v = nc.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
            v[:] = arr
    finally:
        nc.close()
    return {
        "file": str(out.relative_to(ROOT)),
        "shape": {"time": nt, "lat": ny, "lon": nx},
        "vars": ["SST", "SSU", "SSV", "u10", "v10", "swh"],
        "note": "可循环 time_index 调用 nc_to_bgr 抽帧后用 cv2.VideoWriter 或 ffmpeg 合成 mp4",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"generated_by": "scripts/generate_demo_nc_three_modules.py", "modules": {}}
    meta["modules"]["vortex"] = make_eddy(OUT / "mod1_ocean_sst_uv.nc")
    meta["modules"]["wind_wave"] = make_windwave(OUT / "mod2_ocean_wind_wave.nc")
    meta["modules"]["hydro"] = make_hydro_l2(OUT / "mod3_ocean_hydro_L2.nc")
    meta["modules"]["fused_stream_windwave_video"] = make_fused_stream_windwave_video(
        OUT / "mod_fused_stream_windwave_video.nc"
    )
    _write_meta(OUT / "demo_nc_manifest.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
