from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from services.nc_ingest_service import ALLOWED_NC_SUFFIXES, NC_CACHE_DIR
from web_api.deps import REPO_ROOT

router = APIRouter(prefix="/realtime", tags=["realtime"])


def _poll_dir() -> Path:
    raw = os.environ.get("REALTIME_NC_POLL_DIR", "").strip()
    root = Path(raw).expanduser().resolve() if raw else NC_CACHE_DIR.resolve()
    if not root.is_dir():
        raise HTTPException(status_code=503, detail=f"轮询目录不可用: {root}")
    return root


@router.get("/latest")
def latest_nc() -> dict[str, str]:
    """返回目录内最近修改的一份 .nc（占位：准实时接入）。"""
    root = _poll_dir()
    candidates: list[Path] = []
    for pat in ("*.nc", "*.nc4", "*.cdf"):
        candidates.extend(root.glob(pat))
    nc_files = [p for p in candidates if p.is_file() and p.suffix.lower() in ALLOWED_NC_SUFFIXES]
    if not nc_files:
        raise HTTPException(status_code=404, detail="目录内无可用 NetCDF")
    latest = max(nc_files, key=lambda p: p.stat().st_mtime)
    st = latest.stat()
    fingerprint = f"{int(st.st_mtime_ns)}:{st.st_size}"
    abs_latest = latest.resolve()
    try:
        path_str = abs_latest.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        path_str = abs_latest.as_posix()
    return {
        "path": path_str,
        "mtime": str(int(st.st_mtime)),
        "fingerprint": fingerprint,
    }
