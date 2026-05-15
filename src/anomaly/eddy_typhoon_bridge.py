"""从「涡旋/实时」推理结果推断台风联动默认时空窗，并组装 run_detect 所需的 anomaly_result。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.config import load_yaml, resolve_path

COMPANION_NPZ_WIND_KEYS = (
    "demo_wind_observed",
    "demo_wind_predicted",
    "demo_wave_observed",
    "demo_wave_predicted",
)


def safe_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_typhoon_link_yaml_cfgs() -> tuple[dict[str, Any], dict[str, Any]]:
    data_cfg: dict[str, Any] = {}
    demo_cfg: dict[str, Any] = {}
    try:
        raw = load_yaml("config/data.yaml")
        if isinstance(raw, dict):
            data_cfg = raw
    except Exception:
        pass
    try:
        raw = load_yaml("app/config/demo.yaml")
        if isinstance(raw, dict):
            demo_cfg = raw
    except Exception:
        pass
    return data_cfg, demo_cfg


def typhoon_query_bbox_from_configs() -> tuple[float, float, float, float]:
    """
    data.yaml 海区 + demo typhoon_link 外扩，与结果页/联动推断一致（供台风知识库默认表单等复用）。
    """
    data_cfg, demo_cfg = _load_typhoon_link_yaml_cfgs()
    spatial = data_cfg.get("spatial", {}) if isinstance(data_cfg.get("spatial"), dict) else {}
    ty_cfg = demo_cfg.get("typhoon_link", {}) if isinstance(demo_cfg.get("typhoon_link"), dict) else {}
    return _bbox_with_pad_from_spatial(spatial, ty_cfg)


def _resolve_nc_path_for_typhoon_link(eddy_result: dict[str, Any]) -> Path | None:
    """从会话结果 meta 解析当前风浪/涡旋关联的 NetCDF 路径（存在且可读则返回）。"""
    meta = eddy_result.get("meta")
    if not isinstance(meta, dict):
        return None
    raw = meta.get("nc_path")
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    p = Path(s).expanduser()
    p = p.resolve() if p.is_absolute() else resolve_path(s)
    return p if p.is_file() else None


def _time_bounds_from_coord(t_coord: Any) -> tuple[datetime | None, datetime | None]:
    """解析 time 坐标的最小/最大时刻（numpy datetime64、cftime 或 datetime），不依赖 pandas。"""
    arr = np.asarray(t_coord.values).ravel()
    if arr.size < 1:
        return None, None
    if np.issubdtype(arr.dtype, np.datetime64):
        try:
            lo = str(np.datetime_as_string(np.min(arr), unit="s"))
            hi = str(np.datetime_as_string(np.max(arr), unit="s"))
            return datetime.fromisoformat(lo), datetime.fromisoformat(hi)
        except Exception:
            return None, None

    seq = [x for x in arr.tolist() if x is not None]
    if not seq:
        return None, None

    def to_std(dt: object) -> datetime | None:
        if isinstance(dt, datetime):
            return dt
        if not all(hasattr(dt, a) for a in ("year", "month", "day")):
            return None
        try:
            return datetime(
                int(cast(Any, dt).year),
                int(cast(Any, dt).month),
                int(cast(Any, dt).day),
                int(getattr(dt, "hour", 0) or 0),
                int(getattr(dt, "minute", 0) or 0),
                int(getattr(dt, "second", 0) or 0),
            )
        except Exception:
            return None

    parsed = [to_std(x) for x in seq]
    parsed = [x for x in parsed if x is not None]
    if not parsed:
        return None, None
    return min(parsed), max(parsed)


def _read_nc_spatiotemporal_bounds_for_typhoon(nc_path: Path) -> dict[str, Any] | None:
    """
    从 NC 的坐标推断台风联动检索框与时间端点；失败则返回 None（由调用方回退 config）。
    """
    from src.preprocess.netcdf_io import open_netcdf_dataset

    try:
        ds, tmp_copy = open_netcdf_dataset(nc_path)
    except Exception:
        return None

    lat_names = ("lat", "latitude", "LAT", "Latitude", "nav_lat")
    lon_names = ("lon", "longitude", "LON", "Longitude", "nav_lon")
    time_names = ("time", "Time", "TIME")

    try:
        lat_coord = next((ds.coords[n] for n in lat_names if n in ds.coords), None)
        lon_coord = next((ds.coords[n] for n in lon_names if n in ds.coords), None)
        if lat_coord is None or lon_coord is None:
            return None

        lat_vals = np.asarray(lat_coord.values, dtype=np.float64).ravel()
        lon_vals = np.asarray(lon_coord.values, dtype=np.float64).ravel()
        lat_vals = lat_vals[np.isfinite(lat_vals)]
        lon_vals = lon_vals[np.isfinite(lon_vals)]
        if lat_vals.size < 1 or lon_vals.size < 1:
            return None

        lat_min = float(np.min(lat_vals))
        lat_max = float(np.max(lat_vals))
        lon_min = float(np.min(lon_vals))
        lon_max = float(np.max(lon_vals))

        start_dt: datetime | None = None
        end_dt: datetime | None = None
        t_coord = next((ds.coords[n] for n in time_names if n in ds.coords), None)
        if t_coord is not None:
            start_dt, end_dt = _time_bounds_from_coord(t_coord)

        return {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "start_dt": start_dt,
            "end_dt": end_dt,
        }
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)
            except OSError:
                pass


def _bbox_with_pad_from_spatial(
    spatial: dict[str, Any],
    ty_cfg: dict[str, Any],
) -> tuple[float, float, float, float]:
    lon_min = safe_float(spatial.get("lon_min"), 117.0)
    lon_max = safe_float(spatial.get("lon_max"), 127.0)
    lat_min = safe_float(spatial.get("lat_min"), 31.0)
    lat_max = safe_float(spatial.get("lat_max"), 41.0)

    pad_lon = max(0.0, safe_float(ty_cfg.get("query_lon_pad_deg"), 0.0))
    pad_lat = max(0.0, safe_float(ty_cfg.get("query_lat_pad_deg"), 0.0))
    lon_min -= pad_lon
    lon_max += pad_lon
    lat_min -= pad_lat
    lat_max += pad_lat
    lon_min = max(-180.0, min(180.0, lon_min))
    lon_max = max(-180.0, min(180.0, lon_max))
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    lat_min = max(-90.0, min(90.0, lat_min))
    lat_max = max(-90.0, min(90.0, lat_max))
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    return lon_min, lon_max, lat_min, lat_max


def infer_typhoon_link_defaults_from_eddy_result(eddy_result: dict[str, Any]) -> dict[str, Any]:
    """
    与结果页「自动推断」一致：优先用当前会话 NC 的经纬度与时间坐标；否则回退 config/data.yaml 海区 + demo 时间窗。
    """
    data_cfg, demo_cfg = _load_typhoon_link_yaml_cfgs()
    spatial = data_cfg.get("spatial", {}) if isinstance(data_cfg.get("spatial"), dict) else {}
    ty_cfg = demo_cfg.get("typhoon_link", {}) if isinstance(demo_cfg.get("typhoon_link"), dict) else {}

    lon_min, lon_max, lat_min, lat_max = _bbox_with_pad_from_spatial(spatial, ty_cfg)

    generated_at = eddy_result.get("generated_at")
    if isinstance(generated_at, (int, float)):
        end_dt = datetime.fromtimestamp(float(generated_at))
    else:
        end_dt = datetime.now()
    window_hours = int(safe_float(ty_cfg.get("default_window_hours"), 24 * 10))
    start_dt = end_dt - timedelta(hours=max(1, window_hours))

    nc_p = _resolve_nc_path_for_typhoon_link(eddy_result)
    if nc_p is not None:
        bounds = _read_nc_spatiotemporal_bounds_for_typhoon(nc_p)
        if bounds is not None:
            lon_min, lon_max, lat_min, lat_max = _bbox_with_pad_from_spatial(
                {
                    "lon_min": bounds["lon_min"],
                    "lon_max": bounds["lon_max"],
                    "lat_min": bounds["lat_min"],
                    "lat_max": bounds["lat_max"],
                },
                ty_cfg,
            )
            b0 = bounds.get("start_dt")
            b1 = bounds.get("end_dt")
            if isinstance(b0, datetime) and isinstance(b1, datetime):
                start_dt, end_dt = b0, b1
                if start_dt > end_dt:
                    start_dt, end_dt = end_dt, start_dt

    default_top_k = int(safe_float(ty_cfg.get("default_top_k"), 5))
    events_json_path = str(
        ty_cfg.get("events_json_path") or resolve_path("data/processed/anomaly/typhoon_kb/events.json")
    )
    return {
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "top_k": max(1, min(default_top_k, 25)),
        "events_json_path": events_json_path,
    }


def load_wind_wave_companion_from_npz(path: str | Path) -> dict[str, Any] | None:
    """
    读取演示配套 NPZ 中的 demo_wind_* / demo_wave_*（与 scripts/gen_eddy_demo_physics_npz.py 一致）。
    返回可并入 eddy_last_result 的字段；缺键或长度不一致则 None。
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        z = np.load(p)
    except Exception:
        return None
    if not all(k in z.files for k in COMPANION_NPZ_WIND_KEYS):
        return None
    wo = np.asarray(z["demo_wind_observed"], dtype=np.float64).ravel()
    wp = np.asarray(z["demo_wind_predicted"], dtype=np.float64).ravel()
    ho = np.asarray(z["demo_wave_observed"], dtype=np.float64).ravel()
    hp = np.asarray(z["demo_wave_predicted"], dtype=np.float64).ravel()
    n = min(wo.size, wp.size, ho.size, hp.size)
    if n < 1:
        return None
    wo, wp, ho, hp = wo[:n], wp[:n], ho[:n], hp[:n]
    return {
        "demo_wind_observed": wo.astype(float).tolist(),
        "demo_wind_predicted": wp.astype(float).tolist(),
        "demo_wave_observed": ho.astype(float).tolist(),
        "demo_wave_predicted": hp.astype(float).tolist(),
        "wind_wave_from_companion_npz": True,
    }


def apply_wind_wave_companion_to_eddy_result(
    eddy_result: dict[str, Any],
    companion: dict[str, Any] | None,
) -> dict[str, Any]:
    """将配套风浪字段写入成功态 eddy 结果（浅拷贝 + 更新）。"""
    out = dict(eddy_result)
    if companion and out.get("status") == "success":
        out.update(companion)
    return out


def strip_wind_wave_companion_from_eddy_result(eddy_result: dict[str, Any]) -> dict[str, Any]:
    """移除由配套 NPZ 写入的字段（例如用户改传不含 demo_wind_* 的 NPZ 时）。"""
    out = dict(eddy_result)
    for k in COMPANION_NPZ_WIND_KEYS + (
        "wind_wave_from_companion_npz",
        "wind_wave_from_netcdf",
        "wind_wave_assessment_note",
        "wind_wave_nc_extract_meta",
    ):
        out.pop(k, None)
    return out


def _wind_wave_extras_for_anomaly(eddy_result: dict[str, Any]) -> dict[str, Any]:
    """供 compute_anomaly_assessment 使用的残差与尺度（仅配套 NPZ 启用时）。"""
    if not eddy_result.get("wind_wave_from_companion_npz"):
        return {}
    wo = eddy_result.get("demo_wind_observed")
    wp = eddy_result.get("demo_wind_predicted")
    ho = eddy_result.get("demo_wave_observed")
    hp = eddy_result.get("demo_wave_predicted")
    if not (
        isinstance(wo, list)
        and isinstance(wp, list)
        and isinstance(ho, list)
        and isinstance(hp, list)
        and len(wo) == len(wp) == len(ho) == len(hp)
        and len(wo) > 0
    ):
        return {}
    woa = np.asarray(wo, dtype=np.float64)
    wpa = np.asarray(wp, dtype=np.float64)
    hoa = np.asarray(ho, dtype=np.float64)
    hpa = np.asarray(hp, dtype=np.float64)
    wd = woa - wpa
    hd = hoa - hpa
    wr = float(np.mean(wd))
    vr = float(np.mean(hd))
    ws = max(float(np.std(wd)), 0.05)
    vs = max(float(np.std(hd)), 0.02)
    note = eddy_result.get("wind_wave_assessment_note")
    if not (isinstance(note, str) and note.strip()):
        note = "配套 NPZ：演示用风速/波高 obs−pred 序列（非命题方原始格点）。"
    return {
        "wind_residual": wr,
        "wave_residual": vr,
        "wind_mean": 0.0,
        "wind_std": ws,
        "wave_mean": 0.0,
        "wave_std": vs,
        "assessment_note": note.strip(),
    }


def _current_curve_for_detect(eddy_result: dict[str, Any]) -> list[float]:
    timeline = [it for it in (eddy_result.get("timeline") or []) if isinstance(it, dict)]
    tl_scores = [float(it.get("score", 0.0)) for it in timeline]
    if not eddy_result.get("wind_wave_from_companion_npz"):
        return tl_scores
    wo = eddy_result.get("demo_wind_observed")
    wp = eddy_result.get("demo_wind_predicted")
    ho = eddy_result.get("demo_wave_observed")
    hp = eddy_result.get("demo_wave_predicted")
    if not (
        isinstance(wo, list)
        and isinstance(wp, list)
        and isinstance(ho, list)
        and isinstance(hp, list)
        and len(wo) == len(wp) == len(ho) == len(hp)
    ):
        return tl_scores
    woa = np.asarray(wo, dtype=np.float64)
    wpa = np.asarray(wp, dtype=np.float64)
    hoa = np.asarray(ho, dtype=np.float64)
    hpa = np.asarray(hp, dtype=np.float64)
    combo = np.abs(woa - wpa) + np.abs(hoa - hpa)
    nt = len(tl_scores)
    nw = int(combo.shape[0])
    if nt <= 0:
        return combo.astype(float).tolist()
    if nw >= nt:
        return [float(combo[i]) for i in range(nt)]
    out = [float(combo[i]) for i in range(nw)]
    if nw > 0:
        out.extend([float(combo[-1])] * (nt - nw))
    return out


def build_anomaly_result_for_detect(
    eddy_result: dict[str, Any],
    *,
    link_defaults: dict[str, Any],
) -> dict[str, Any]:
    """组装 run_detect 的 anomaly_result（时间窗与海区来自 link_defaults；配套 NPZ 时写入风浪残差与曲线）。"""
    ar: dict[str, Any] = {
        "start_time": link_defaults["start_time"],
        "end_time": link_defaults["end_time"],
        "lon_min": float(link_defaults["lon_min"]),
        "lon_max": float(link_defaults["lon_max"]),
        "lat_min": float(link_defaults["lat_min"]),
        "lat_max": float(link_defaults["lat_max"]),
        "peak_score": eddy_result.get("peak_score"),
        "current_curve": _current_curve_for_detect(eddy_result),
    }
    ar.update(_wind_wave_extras_for_anomaly(eddy_result))
    return ar
