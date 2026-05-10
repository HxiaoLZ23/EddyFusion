from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.typhoon_kb import QueryBox, query_typhoon_events
from src.utils.config import resolve_path


def _parse_time(text: str) -> datetime:
    fmts = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S")
    raw = text.strip()
    for f in fmts:
        try:
            return datetime.strptime(raw, f)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {text}")


def _export_rows(rows: list[dict[str, Any]], export_path: str) -> Path:
    p = resolve_path(export_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
    if p.suffix.lower() == ".csv":
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = [
                "event_id",
                "name",
                "start_time",
                "end_time",
                "intensity_level",
                "peak_wind_kt",
                "bbox_overlap_ratio",
                "time_overlap_hours",
                "score",
                "summary",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in fieldnames})
        return p
    raise ValueError("export 文件后缀需为 .json 或 .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="按时间窗+区域查询台风知识库")
    parser.add_argument("--events-json", type=str, default="data/processed/anomaly/typhoon_kb/events.json")
    parser.add_argument("--start-time", type=str, required=True)
    parser.add_argument("--end-time", type=str, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--export", type=str, default="")
    args = parser.parse_args()

    q = QueryBox(
        start_time=_parse_time(args.start_time),
        end_time=_parse_time(args.end_time),
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
    )
    rows = query_typhoon_events(events_json_path=args.events_json, query=q, top_k=args.top_k)
    print(f"matched events: {len(rows)}")
    for i, r in enumerate(rows[: min(10, len(rows))], start=1):
        print(
            f"{i:02d}. {r['event_id']} | {r.get('name', '')} | "
            f"score={r['score']} | overlap_h={r['time_overlap_hours']} | bbox={r['bbox_overlap_ratio']}"
        )
    if args.export:
        out = _export_rows(rows, args.export)
        print(f"exported: {out}")


if __name__ == "__main__":
    main()
