from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.typhoon_kb import QueryBox, query_typhoon_events
from src.utils.config import resolve_path


def _run_case(
    *,
    case_id: str,
    events_json: str,
    start_time: str,
    end_time: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    top_k: int = 5,
) -> dict:
    from datetime import datetime

    q = QueryBox(
        start_time=datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"),
        end_time=datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"),
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )
    rows = query_typhoon_events(events_json_path=events_json, query=q, top_k=top_k)
    return {
        "case_id": case_id,
        "query": {
            "start_time": start_time,
            "end_time": end_time,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        },
        "results": rows,
    }


def main() -> None:
    events_json = "data/processed/anomaly/typhoon_kb/events.json"
    cases = [
        _run_case(
            case_id="case_01_demo_link",
            events_json=events_json,
            start_time="2024-08-01 00:00:00",
            end_time="2024-08-10 23:59:59",
            lon_min=117.0,
            lon_max=127.0,
            lat_min=31.0,
            lat_max=41.0,
            top_k=5,
        ),
        _run_case(
            case_id="case_02_demo_link",
            events_json=events_json,
            start_time="2025-07-01 00:00:00",
            end_time="2025-07-10 23:59:59",
            lon_min=117.0,
            lon_max=127.0,
            lat_min=31.0,
            lat_max=41.0,
            top_k=5,
        ),
    ]
    out = resolve_path("data/processed/anomaly/typhoon_kb/demo_cases.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    md = resolve_path("data/processed/anomaly/typhoon_kb/demo_cases.md")
    lines = ["# 台风知识库联动案例", ""]
    for c in cases:
        lines.append(f"## {c['case_id']}")
        q = c["query"]
        lines.append(
            f"- query: {q['start_time']} ~ {q['end_time']} | "
            f"lon[{q['lon_min']},{q['lon_max']}] lat[{q['lat_min']},{q['lat_max']}]"
        )
        if c["results"]:
            for r in c["results"][:3]:
                lines.append(f"- hit: {r['event_id']} {r.get('name','')} score={r['score']}")
        else:
            lines.append("- hit: (none)")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
