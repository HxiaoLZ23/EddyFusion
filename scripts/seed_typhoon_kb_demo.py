"""写入演示用台风事件索引（无需 IBTrACS），供离线风浪/涡旋联动与 Web 二级页验收。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_path

# 覆盖 demo NC 常见时空：1993 涡旋样例、2015 风浪 merged、西北太平洋
DEMO_EVENTS: list[dict] = [
    {
        "event_id": "DEMO_199308_WP",
        "name": "DEMO_WP_1993",
        "season": "1993",
        "basin": "WP",
        "start_time": "1993-08-01 00:00:00",
        "end_time": "1993-08-31 23:59:59",
        "center_lon": 130.0,
        "center_lat": 25.0,
        "lon_min": 115.0,
        "lon_max": 145.0,
        "lat_min": 12.0,
        "lat_max": 38.0,
        "peak_wind_kt": 85.0,
        "intensity_level": "typhoon",
        "n_points": 48,
        "retrieval_keys": ["time:1993-08", "grid:26:5", "level:typhoon"],
        "wind_track_kt": [30.0, 45.0, 60.0, 75.0, 85.0, 70.0, 55.0],
        "wind_track_mps": [15.4333, 23.15, 30.8667, 38.5833, 43.7277, 36.0111, 28.2944],
        "track_times": [
            "1993-08-05 00:00:00",
            "1993-08-08 00:00:00",
            "1993-08-11 00:00:00",
            "1993-08-14 00:00:00",
            "1993-08-17 00:00:00",
            "1993-08-20 00:00:00",
            "1993-08-23 00:00:00",
        ],
        "series_source": "demo_synthetic",
    },
    {
        "event_id": "DEMO_201501_WP",
        "name": "DEMO_WP_2015",
        "season": "2015",
        "basin": "WP",
        "start_time": "2015-01-01 00:00:00",
        "end_time": "2015-01-31 23:59:59",
        "center_lon": 128.0,
        "center_lat": 22.0,
        "lon_min": 112.0,
        "lon_max": 148.0,
        "lat_min": 10.0,
        "lat_max": 36.0,
        "peak_wind_kt": 72.0,
        "intensity_level": "typhoon",
        "n_points": 40,
        "retrieval_keys": ["time:2015-01", "grid:25:4", "level:typhoon"],
        "wind_track_kt": [22.0, 28.0, 38.0, 52.0, 65.0, 72.0, 68.0, 55.0, 40.0],
        "wind_track_mps": [11.3178, 14.4044, 19.5489, 26.7511, 33.4389, 37.04, 34.9822, 28.2944, 20.5778],
        "track_times": [
            "2015-01-03 00:00:00",
            "2015-01-05 00:00:00",
            "2015-01-07 00:00:00",
            "2015-01-09 00:00:00",
            "2015-01-11 00:00:00",
            "2015-01-13 00:00:00",
            "2015-01-15 00:00:00",
            "2015-01-17 00:00:00",
            "2015-01-19 00:00:00",
        ],
        "series_source": "demo_synthetic",
    },
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
        "wind_track_kt": [35.0, 40.0, 45.0, 50.0, 55.0],
        "wind_track_mps": [round(35.0 * 0.514444, 4), round(40.0 * 0.514444, 4), round(45.0 * 0.514444, 4), round(50.0 * 0.514444, 4), round(55.0 * 0.514444, 4)],
        "track_times": [
            "2024-08-01 00:00:00",
            "2024-08-03 00:00:00",
            "2024-08-06 00:00:00",
            "2024-08-09 00:00:00",
            "2024-08-12 00:00:00",
        ],
        "series_source": "ibtracs_center_wind",
    },
]


def main() -> None:
    events_json = resolve_path("data/processed/anomaly/typhoon_kb/events.json")
    events_csv = resolve_path("data/processed/anomaly/typhoon_kb/events.csv")
    retrieval_json = resolve_path("data/processed/anomaly/typhoon_kb/retrieval_index.json")
    typhoon_index = resolve_path("data/processed/anomaly/typhoon_index.json")
    for p in (events_json, events_csv, retrieval_json, typhoon_index):
        p.parent.mkdir(parents=True, exist_ok=True)

    events_json.write_text(json.dumps(DEMO_EVENTS, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
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
    ]
    with events_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in DEMO_EVENTS:
            row = dict(e)
            row["retrieval_keys"] = "|".join(e["retrieval_keys"])
            w.writerow(row)

    retrieval_index: dict[str, list[str]] = {}
    for e in DEMO_EVENTS:
        for k in e["retrieval_keys"]:
            retrieval_index.setdefault(k, []).append(e["event_id"])
    retrieval_json.write_text(
        json.dumps({"retrieval_index": retrieval_index, "events_path": str(events_json)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typhoon_index.write_text(
        json.dumps(
            {
                "source": "demo_seed",
                "version": "local",
                "events_count": len(DEMO_EVENTS),
                "events_json": str(events_json),
                "events_csv": str(events_csv),
                "retrieval_json": str(retrieval_json),
                "note": "演示索引；完整库请运行 scripts/run_typhoon_kb.ps1 + IBTrACS",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(DEMO_EVENTS)} demo events -> {events_json}")


if __name__ == "__main__":
    main()
