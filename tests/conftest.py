from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def demo_events_json(tmp_path: Path) -> Path:
    events = [
        {
            "event_id": "DEMO_202408_WP",
            "name": "DEMO_WP_2024",
            "season": "2024",
            "basin": "WP",
            "start_time": "2024-08-01 00:00:00",
            "end_time": "2024-08-15 23:59:59",
            "center_lon": 122.0,
            "center_lat": 32.0,
            "lon_min": 117.0,
            "lon_max": 127.0,
            "lat_min": 28.0,
            "lat_max": 38.0,
            "peak_wind_kt": 55.0,
            "intensity_level": "tropical_storm",
            "n_points": 24,
            "retrieval_keys": ["time:2024-08", "grid:24:6", "level:tropical_storm"],
            "wind_track_mps": [20.0, 22.0, 25.0, 28.0],
            "series_source": "ibtracs_center_wind",
        },
        {
            "event_id": "DEMO_202401_IO",
            "name": "DEMO_IO_2024",
            "season": "2024",
            "basin": "IO",
            "start_time": "2024-01-01 00:00:00",
            "end_time": "2024-01-10 23:59:59",
            "center_lon": 80.0,
            "center_lat": -12.0,
            "lon_min": 70.0,
            "lon_max": 90.0,
            "lat_min": -20.0,
            "lat_max": -5.0,
            "peak_wind_kt": 72.0,
            "intensity_level": "typhoon",
            "n_points": 20,
            "retrieval_keys": ["time:2024-01", "grid:16:-3", "level:typhoon"],
        },
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def demo_windwave_nc() -> Path:
    from netCDF4 import Dataset

    nc_dir = REPO_ROOT / "app" / "data" / "nc_uploads"
    nc_dir.mkdir(parents=True, exist_ok=True)
    path = nc_dir / f"system_test_windwave_{uuid.uuid4().hex[:8]}.nc"
    nt, ny, nx = 12, 6, 6

    nc_target = path.relative_to(REPO_ROOT).as_posix()
    with Dataset(nc_target, "w", format="NETCDF4") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("lat", ny)
        ds.createDimension("lon", nx)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "hours since 2024-08-01 00:00:00"
        time.calendar = "standard"
        time[:] = np.arange(nt, dtype=np.float64) * 6.0

        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        lat[:] = np.linspace(28.0, 38.0, ny, dtype=np.float32)
        lon[:] = np.linspace(117.0, 127.0, nx, dtype=np.float32)

        u10 = ds.createVariable("u10", "f4", ("time", "lat", "lon"))
        v10 = ds.createVariable("v10", "f4", ("time", "lat", "lon"))
        swh = ds.createVariable("swh", "f4", ("time", "lat", "lon"))

        t = np.arange(nt, dtype=np.float32).reshape(nt, 1, 1)
        yy = np.linspace(-1.0, 1.0, ny, dtype=np.float32).reshape(1, ny, 1)
        xx = np.linspace(-1.0, 1.0, nx, dtype=np.float32).reshape(1, 1, nx)
        u10[:] = 8.0 + 0.25 * t + 0.1 * xx
        v10[:] = 4.0 + 0.12 * t + 0.1 * yy
        swh[:] = 1.4 + 0.04 * t + 0.03 * (xx + yy)

    try:
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@pytest.fixture()
def demo_eddy_nc() -> Path:
    """合成 3ch 涡旋演示 NC（ADT + U + V）。"""
    from netCDF4 import Dataset

    nc_dir = REPO_ROOT / "app" / "data" / "nc_uploads"
    nc_dir.mkdir(parents=True, exist_ok=True)
    path = nc_dir / f"system_test_eddy_{uuid.uuid4().hex[:8]}.nc"
    nt, ny, nx = 8, 10, 12

    nc_target = path.relative_to(REPO_ROOT).as_posix()
    with Dataset(nc_target, "w", format="NETCDF4") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("lat", ny)
        ds.createDimension("lon", nx)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "hours since 2024-06-01 00:00:00"
        time.calendar = "standard"
        time[:] = np.arange(nt, dtype=np.float64) * 6.0

        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        lat[:] = np.linspace(20.0, 35.0, ny, dtype=np.float32)
        lon[:] = np.linspace(115.0, 130.0, nx, dtype=np.float32)

        adt = ds.createVariable("adt", "f4", ("time", "lat", "lon"))
        ugos = ds.createVariable("ugos", "f4", ("time", "lat", "lon"))
        vgos = ds.createVariable("vgos", "f4", ("time", "lat", "lon"))

        t = np.arange(nt, dtype=np.float32).reshape(nt, 1, 1)
        yy = np.linspace(-1.0, 1.0, ny, dtype=np.float32).reshape(1, ny, 1)
        xx = np.linspace(-1.0, 1.0, nx, dtype=np.float32).reshape(1, 1, nx)
        adt[:] = 0.55 + 0.02 * t + 0.01 * (xx + yy)
        ugos[:] = -0.15 + 0.01 * t + 0.02 * yy
        vgos[:] = 0.08 + 0.008 * t - 0.02 * xx

    try:
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def eddy_weights_available() -> bool:
    """本地是否存在可加载的 3ch 涡旋权重（T4 全链路需要）。"""
    from app.services.eddy_demo_service import default_eddy_weight_path_for_stack
    from src.utils.config import resolve_path

    return resolve_path(default_eddy_weight_path_for_stack("3ch")).is_file()


def rel_repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()
