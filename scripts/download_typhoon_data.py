from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_path


IBTRACS_DEFAULT_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="下载或登记台风公开数据源（IBTrACS/CMA）")
    parser.add_argument("--source", choices=("ibtracs", "cma"), default="ibtracs")
    parser.add_argument("--output", type=str, default="data/raw/typhoon/ibtracs/ibtracs.ALL.list.v04r01.csv")
    parser.add_argument("--url", type=str, default=IBTRACS_DEFAULT_URL)
    parser.add_argument(
        "--local-file",
        type=str,
        default="",
        help="source=cma 时可传本地文件路径，仅做登记与复制，不做网络下载",
    )
    args = parser.parse_args()

    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "ibtracs":
        print(f"downloading: {args.url}")
        urllib.request.urlretrieve(args.url, out)
        method = "http_download"
        url = args.url
    else:
        if not args.local_file:
            raise SystemExit("source=cma 时需提供 --local-file")
        src = resolve_path(args.local_file)
        if not src.is_file():
            raise FileNotFoundError(f"本地文件不存在: {src}")
        out.write_bytes(src.read_bytes())
        method = "local_copy"
        url = "manual"

    meta = {
        "source": args.source,
        "method": method,
        "url": url,
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_path": str(out),
        "sha256": _sha256(out),
    }
    meta_path = out.parent / "source_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote data file: {out}")
    print(f"wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
