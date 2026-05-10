from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.typhoon_kb import build_typhoon_index
from src.utils.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="构建台风知识库事件索引（json/csv + 检索键）")
    parser.add_argument("--source-csv", type=str, default="data/raw/typhoon/ibtracs/ibtracs.ALL.list.v04r01.csv")
    parser.add_argument("--source-name", type=str, default="IBTrACS")
    parser.add_argument("--source-version", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--events-json", type=str, default="data/processed/anomaly/typhoon_kb/events.json")
    parser.add_argument("--events-csv", type=str, default="data/processed/anomaly/typhoon_kb/events.csv")
    parser.add_argument("--retrieval-json", type=str, default="data/processed/anomaly/typhoon_kb/retrieval_index.json")
    parser.add_argument("--typhoon-index", type=str, default="data/processed/anomaly/typhoon_index.json")
    args = parser.parse_args()

    summary = build_typhoon_index(
        source_csv_path=args.source_csv,
        output_events_json=args.events_json,
        output_events_csv=args.events_csv,
        output_retrieval_json=args.retrieval_json,
        output_typhoon_index_json=args.typhoon_index,
        source_name=args.source_name,
        source_version=args.source_version,
    )
    print("typhoon kb built")
    for k in ("source", "version", "events_count", "events_json", "events_csv", "retrieval_json"):
        print(f"- {k}: {summary.get(k)}")
    print(f"- typhoon_index: {resolve_path(args.typhoon_index)}")


if __name__ == "__main__":
    main()
