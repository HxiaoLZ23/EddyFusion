from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _list_nc_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pat in ("*.nc", "*.nc4", "*.cdf"):
        candidates.extend(root.glob(pat))
    return [p for p in candidates if p.is_file() and p.suffix.lower() in ALLOWED_NC_SUFFIXES]


def _path_token(p: Path) -> str:
    abs_p = p.resolve()
    try:
        return abs_p.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return abs_p.as_posix()


def _file_info(p: Path) -> dict[str, Any]:
    st = p.stat()
    mtime = int(st.st_mtime)
    age_sec = max(0, int(time.time()) - mtime)
    return {
        "path": _path_token(p),
        "filename": p.name,
        "mtime": str(mtime),
        "mtime_iso": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": int(st.st_size),
        "size_mb": round(st.st_size / (1024**2), 4),
        "fingerprint": f"{int(st.st_mtime_ns)}:{st.st_size}",
        "age_sec": age_sec,
        "stale": age_sec > int(os.environ.get("REALTIME_NC_STALE_SEC", "3600")),
    }


@router.get("/status")
def realtime_status() -> dict[str, Any]:
    """
    G15：准实时连接器状态 — 轮询目录、文件数、最新文件摘要、环境配置。
    """
    try:
        root = _poll_dir()
    except HTTPException as e:
        return {
            "connected": False,
            "ready": False,
            "error": str(e.detail),
            "poll_dir": os.environ.get("REALTIME_NC_POLL_DIR") or str(NC_CACHE_DIR),
            "source": "local_directory_poll",
        }

    files = _list_nc_files(root)
    latest_info = None
    if files:
        latest = max(files, key=lambda p: p.stat().st_mtime)
        latest_info = _file_info(latest)

    poll_dir_display = str(root)
    try:
        poll_dir_display = root.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        pass

    return {
        "connected": True,
        "ready": bool(files),
        "source": "local_directory_poll",
        "poll_dir": poll_dir_display,
        "poll_interval_hint_sec": int(os.environ.get("REALTIME_NC_POLL_INTERVAL", "30")),
        "stale_threshold_sec": int(os.environ.get("REALTIME_NC_STALE_SEC", "3600")),
        "nc_count": len(files),
        "latest": latest_info,
        "checked_at_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/latest")
def latest_nc() -> dict[str, Any]:
    """返回目录内最近修改的一份 NC（G15 强化：含体积、时效、陈旧标记）。"""
    root = _poll_dir()
    nc_files = _list_nc_files(root)
    if not nc_files:
        raise HTTPException(status_code=404, detail="目录内无可用 NetCDF")
    latest = max(nc_files, key=lambda p: p.stat().st_mtime)
    info = _file_info(latest)
    return {
        "path": info["path"],
        "filename": info["filename"],
        "mtime": info["mtime"],
        "mtime_iso": info["mtime_iso"],
        "fingerprint": info["fingerprint"],
        "size_bytes": info["size_bytes"],
        "size_mb": info["size_mb"],
        "age_sec": info["age_sec"],
        "stale": info["stale"],
    }
