from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import cv2

def _project_root() -> Path:
    # app/services/*.py -> app/services -> app -> repo root
    return Path(__file__).resolve().parents[2]

MEDIA_DIR = _project_root() / "app" / "data" / "media"
MAX_UPLOAD_MB = 500
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def allowed_suffixes_text() -> str:
    return ", ".join(sorted(ALLOWED_SUFFIXES))


def save_uploaded_video(uploaded_file: Any) -> tuple[Path, str]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"不支持的文件格式: {suffix}")

    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024**2)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(f"文件过大: {size_mb:.1f}MB，建议不超过 {MAX_UPLOAD_MB}MB")

    task_id = uuid.uuid4().hex[:12]
    digest = hashlib.sha1(file_bytes).hexdigest()[:8]
    safe_name = f"{task_id}_{digest}{suffix}"
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    path = MEDIA_DIR / safe_name
    path.write_bytes(file_bytes)
    return path, task_id


def extract_video_metadata(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    meta: dict[str, Any] = {
        "path": str(p),
        "size_mb": p.stat().st_size / (1024**2) if p.exists() else 0.0,
        "resolution": "unknown",
        "fps": "unknown",
        "frame_count": "unknown",
        "duration_sec": "unknown",
    }
    if not p.exists():
        return meta

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return meta
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
        duration = frame_count / fps if fps > 0 else 0.0
        meta.update(
            {
                "resolution": f"{int(width)}x{int(height)}" if width > 0 and height > 0 else "unknown",
                "fps": round(float(fps), 3) if fps > 0 else "unknown",
                "frame_count": int(frame_count) if frame_count > 0 else "unknown",
                "duration_sec": round(float(duration), 3) if duration > 0 else "unknown",
            }
        )
    finally:
        cap.release()
    return meta


def cleanup_old_media(max_files: int = 20) -> None:
    if not MEDIA_DIR.exists():
        return
    files = [p for p in MEDIA_DIR.iterdir() if p.is_file()]
    if len(files) <= max_files:
        return
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for fp in files[max_files:]:
        try:
            os.remove(fp)
        except OSError:
            continue

