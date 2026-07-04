"""将演示台风事件追加进已有 events.json（不覆盖全量 IBTrACS 索引）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.seed_typhoon_kb_demo import DEMO_EVENTS  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402


def merge_demo_events(*, events_json: str | Path | None = None) -> dict[str, int]:
    path = resolve_path(events_json or "data/processed/anomaly/typhoon_kb/events.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"events.json 应为数组: {path}")
    else:
        rows = []

    existing = {str(r.get("event_id")) for r in rows if isinstance(r, dict) and r.get("event_id")}
    added = 0
    for event in DEMO_EVENTS:
        eid = str(event["event_id"])
        if eid in existing:
            continue
        rows.append(dict(event))
        existing.add(eid)
        added += 1

    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"total": len(rows), "added": added, "events_json": str(path)}


def main() -> None:
    stats = merge_demo_events()
    print(
        f"merged demo typhoon events: +{stats['added']} (total {stats['total']}) -> {stats['events_json']}"
    )


if __name__ == "__main__":
    main()
