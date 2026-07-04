from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import resolve_path


def _parse_time(text: str) -> datetime | None:
    raw = (text or "").strip()
    if not raw:
        return None
    # 常见 IBTrACS: "YYYY-MM-DD HH:MM:SS"
    fmts = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S")
    for f in fmts:
        try:
            return datetime.strptime(raw, f)
        except ValueError:
            continue
    return None


def _to_float(text: str | None, default: float = 0.0) -> float:
    if text is None:
        return default
    t = str(text).strip()
    if t == "" or t.upper() in {"NA", "NAN"}:
        return default
    try:
        v = float(t)
    except ValueError:
        return default
    if v <= -900:  # IBTrACS 缺测常填 -999
        return default
    return v


# IBTrACS 多机构风速列；按优先级取首个有效值（勿用 `WMO or USA`：空格 WMO 会阻断 USA）
_WIND_KT_COLUMN_PRIORITY: tuple[str, ...] = (
    "WMO_WIND",
    "USA_WIND",
    "TOKYO_WIND",
    "CMA_WIND",
    "HKO_WIND",
    "KMA_WIND",
    "BOM_WIND",
    "NEUMANN_WIND",
    "TD9636_WIND",
    "TD9635_WIND",
    "DS824_WIND",
    "REUNION_WIND",
    "NADI_WIND",
    "WELLINGTON_WIND",
)


def _wind_kt_from_row(row: dict[str, str]) -> float:
    """从 IBTrACS 行解析风速 (kt)；跳过空白/缺测，支持多机构回退。"""
    for key in _WIND_KT_COLUMN_PRIORITY:
        v = _to_float(row.get(key), default=-1.0)
        if v >= 0.0:
            return float(v)
    return 0.0


KT_TO_MPS = 0.514444


def _wind_level(wind_kt: float) -> str:
    # 简化等级：便于展示和检索，不作为气象标准发布口径
    if wind_kt >= 64:
        return "typhoon"
    if wind_kt >= 34:
        return "tropical_storm"
    if wind_kt > 0:
        return "tropical_depression"
    return "unknown"


def _time_overlap_hours(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> float:
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    delta = (e - s).total_seconds() / 3600.0
    return max(0.0, delta)


def _query_center(query: QueryBox) -> tuple[float, float]:
    return (
        (float(query.lon_min) + float(query.lon_max)) / 2.0,
        (float(query.lat_min) + float(query.lat_max)) / 2.0,
    )


def _center_distance_deg(q_lon: float, q_lat: float, event_lon: float, event_lat: float) -> float:
    dlon = float(event_lon) - float(q_lon)
    dlat = float(event_lat) - float(q_lat)
    return float((dlon * dlon + dlat * dlat) ** 0.5)


def _prescreen_score(*, bbox_ratio: float, center_distance_deg: float, mode: str) -> float:
    """初筛排序分（不含持续时间）；最终 Top-K 仍由 DTW 决定。"""
    m = str(mode or "bbox_ratio").strip().lower()
    if m == "center_distance":
        return 1.0 / max(float(center_distance_deg), 1e-6)
    return float(bbox_ratio)


def _load_prescreen_score_mode() -> str:
    try:
        from src.anomaly.eddy_typhoon_bridge import _load_typhoon_link_yaml_cfgs

        _, demo_cfg = _load_typhoon_link_yaml_cfgs()
        ty_cfg = demo_cfg.get("typhoon_link") if isinstance(demo_cfg.get("typhoon_link"), dict) else {}
        if isinstance(ty_cfg, dict):
            raw = ty_cfg.get("prescreen_score_mode")
            if raw is not None and str(raw).strip():
                return str(raw).strip().lower()
    except Exception:
        pass
    return "bbox_ratio"


def _bbox_overlap_ratio(
    *,
    event_lon_min: float,
    event_lon_max: float,
    event_lat_min: float,
    event_lat_max: float,
    q_lon_min: float,
    q_lon_max: float,
    q_lat_min: float,
    q_lat_max: float,
) -> float:
    ix = max(0.0, min(event_lon_max, q_lon_max) - max(event_lon_min, q_lon_min))
    iy = max(0.0, min(event_lat_max, q_lat_max) - max(event_lat_min, q_lat_min))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    q_area = max(1e-6, (q_lon_max - q_lon_min) * (q_lat_max - q_lat_min))
    return float(inter / q_area)


@dataclass(frozen=True)
class QueryBox:
    start_time: datetime
    end_time: datetime
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def _row_event_id(row: dict[str, str], idx: int) -> str:
    sid = (row.get("SID") or row.get("sid") or "").strip()
    season = (row.get("SEASON") or row.get("season") or "").strip()
    name = (row.get("NAME") or row.get("name") or "").strip()
    if sid:
        return sid
    if season and name:
        return f"{season}_{name}".replace(" ", "_")
    return f"event_{idx:06d}"


def build_typhoon_index(
    *,
    source_csv_path: str | Path,
    output_events_json: str | Path,
    output_events_csv: str | Path,
    output_retrieval_json: str | Path,
    output_typhoon_index_json: str | Path,
    source_name: str,
    source_version: str,
) -> dict[str, Any]:
    src = resolve_path(source_csv_path)
    if not src.is_file():
        raise FileNotFoundError(f"台风源文件不存在: {src}")

    by_event: dict[str, list[dict[str, Any]]] = {}
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            t = _parse_time(row.get("ISO_TIME", ""))
            if t is None:
                continue
            event_id = _row_event_id(row, i)
            lon = _to_float(row.get("LON"), 0.0)
            lat = _to_float(row.get("LAT"), 0.0)
            wind = _wind_kt_from_row(row)
            rec = {
                "time": t,
                "lon": lon,
                "lat": lat,
                "wind_kt": wind,
                "name": (row.get("NAME") or "").strip(),
                "basin": (row.get("BASIN") or "").strip(),
                "season": (row.get("SEASON") or "").strip(),
            }
            by_event.setdefault(event_id, []).append(rec)

    events: list[dict[str, Any]] = []
    retrieval_index: dict[str, list[str]] = {}
    for event_id, recs in by_event.items():
        recs = sorted(recs, key=lambda x: x["time"])
        st = recs[0]["time"]
        ed = recs[-1]["time"]
        lon_min = min(float(r["lon"]) for r in recs)
        lon_max = max(float(r["lon"]) for r in recs)
        lat_min = min(float(r["lat"]) for r in recs)
        lat_max = max(float(r["lat"]) for r in recs)
        peak = max(float(r["wind_kt"]) for r in recs)
        level = _wind_level(peak)
        center_lon = sum(float(r["lon"]) for r in recs) / max(1, len(recs))
        center_lat = sum(float(r["lat"]) for r in recs) / max(1, len(recs))
        name = next((str(r["name"]) for r in recs if str(r["name"])), "")
        basin = next((str(r["basin"]) for r in recs if str(r["basin"])), "")
        season = next((str(r["season"]) for r in recs if str(r["season"])), "")

        # anomaly 可对接检索键：时间窗(月) + 空间格网(5度)
        month_key = st.strftime("%Y-%m")
        grid_lon = int(center_lon // 5)
        grid_lat = int(center_lat // 5)
        retrieval_keys = [f"time:{month_key}", f"grid:{grid_lon}:{grid_lat}", f"level:{level}"]
        for k in retrieval_keys:
            retrieval_index.setdefault(k, []).append(event_id)

        wind_track_kt = [round(float(r["wind_kt"]), 3) for r in recs]
        track_times = [r["time"].strftime("%Y-%m-%d %H:%M:%S") for r in recs]
        wind_track_mps = [round(v * KT_TO_MPS, 4) for v in wind_track_kt]

        events.append(
            {
                "event_id": event_id,
                "name": name,
                "season": season,
                "basin": basin,
                "start_time": st.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": ed.strftime("%Y-%m-%d %H:%M:%S"),
                "center_lon": center_lon,
                "center_lat": center_lat,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "peak_wind_kt": peak,
                "intensity_level": level,
                "n_points": len(recs),
                "retrieval_keys": retrieval_keys,
                "wind_track_kt": wind_track_kt,
                "wind_track_mps": wind_track_mps,
                "track_times": track_times,
                "series_source": "ibtracs_center_wind",
            }
        )

    events = sorted(events, key=lambda x: (x["start_time"], x["event_id"]))

    out_events_json = resolve_path(output_events_json)
    out_events_csv = resolve_path(output_events_csv)
    out_retrieval = resolve_path(output_retrieval_json)
    out_typhoon_index = resolve_path(output_typhoon_index_json)
    for p in (out_events_json, out_events_csv, out_retrieval, out_typhoon_index):
        p.parent.mkdir(parents=True, exist_ok=True)

    out_events_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_events_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "name",
                "season",
                "basin",
                "start_time",
                "end_time",
                "center_lon",
                "center_lat",
                "lon_min",
                "lon_max",
                "lat_min",
                "lat_max",
                "peak_wind_kt",
                "intensity_level",
                "n_points",
                "retrieval_keys",
            ],
        )
        writer.writeheader()
        for e in events:
            row = {k: v for k, v in e.items() if k in writer.fieldnames}
            row["retrieval_keys"] = "|".join(e["retrieval_keys"])
            writer.writerow(row)

    out_retrieval.write_text(
        json.dumps({"retrieval_index": retrieval_index, "events_path": str(out_events_json)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typhoon_index = {
        "source": source_name,
        "version": source_version,
        "events_count": len(events),
        "events_json": str(out_events_json),
        "events_csv": str(out_events_csv),
        "retrieval_json": str(out_retrieval),
        "fields": [
            "event_id",
            "start_time",
            "end_time",
            "center_lon",
            "center_lat",
            "peak_wind_kt",
            "intensity_level",
            "retrieval_keys",
        ],
    }
    out_typhoon_index.write_text(json.dumps(typhoon_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return typhoon_index


def query_typhoon_events(
    *,
    events_json_path: str | Path,
    query: QueryBox,
    top_k: int = 20,
    prescreen_score_mode: str | None = None,
) -> list[dict[str, Any]]:
    events_path = resolve_path(events_json_path)
    if not events_path.is_file():
        raise FileNotFoundError(f"事件索引不存在: {events_path}")
    events = json.loads(events_path.read_text(encoding="utf-8"))
    score_mode = str(prescreen_score_mode or _load_prescreen_score_mode()).strip().lower()
    q_lon_c, q_lat_c = _query_center(query)
    out: list[dict[str, Any]] = []
    for e in events:
        st = _parse_time(str(e.get("start_time", "")))
        ed = _parse_time(str(e.get("end_time", "")))
        if st is None or ed is None:
            continue
        overlap_h = _time_overlap_hours(st, ed, query.start_time, query.end_time)
        if overlap_h <= 0:
            continue
        bbox_ratio = _bbox_overlap_ratio(
            event_lon_min=float(e.get("lon_min", 0.0)),
            event_lon_max=float(e.get("lon_max", 0.0)),
            event_lat_min=float(e.get("lat_min", 0.0)),
            event_lat_max=float(e.get("lat_max", 0.0)),
            q_lon_min=query.lon_min,
            q_lon_max=query.lon_max,
            q_lat_min=query.lat_min,
            q_lat_max=query.lat_max,
        )
        if bbox_ratio <= 0:
            continue
        center_lon = float(e.get("center_lon", (float(e.get("lon_min", 0)) + float(e.get("lon_max", 0))) / 2.0))
        center_lat = float(e.get("center_lat", (float(e.get("lat_min", 0)) + float(e.get("lat_max", 0))) / 2.0))
        center_dist = _center_distance_deg(q_lon_c, q_lat_c, center_lon, center_lat)
        score = _prescreen_score(
            bbox_ratio=bbox_ratio,
            center_distance_deg=center_dist,
            mode=score_mode,
        )
        row: dict[str, Any] = {
            "event_id": e.get("event_id"),
            "name": e.get("name", ""),
            "start_time": e.get("start_time"),
            "end_time": e.get("end_time"),
            "intensity_level": e.get("intensity_level"),
            "peak_wind_kt": e.get("peak_wind_kt"),
            "bbox_overlap_ratio": round(bbox_ratio, 6),
            "time_overlap_hours": round(overlap_h, 3),
            "center_distance_deg": round(center_dist, 6),
            "prescreen_score_mode": score_mode,
            "score": round(score, 6),
            "summary": f"{e.get('event_id')}({e.get('name', '')}) {e.get('start_time')}~{e.get('end_time')}",
        }
        track_mps = e.get("wind_track_mps")
        track_kt = e.get("wind_track_kt")
        if isinstance(track_mps, list) and track_mps:
            row["wind_track_mps"] = [float(v) for v in track_mps]
            row["series_source"] = e.get("series_source", "ibtracs_center_wind")
        elif isinstance(track_kt, list) and track_kt:
            row["wind_track_kt"] = [float(v) for v in track_kt]
            row["wind_track_mps"] = [float(v) * KT_TO_MPS for v in track_kt]
            row["series_source"] = e.get("series_source", "ibtracs_center_wind")
        out.append(row)
    out.sort(key=lambda x: (-float(x["score"]), str(x["event_id"])))
    return out[: max(1, int(top_k))]
