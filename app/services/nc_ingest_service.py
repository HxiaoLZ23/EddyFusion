"""NetCDF 上传落盘与轻量元数据摘要（演示/准实时入口前置）。"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

MAX_NC_UPLOAD_MB = 500
ALLOWED_NC_SUFFIXES = {".nc", ".nc4", ".cdf"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


NC_CACHE_DIR = _project_root() / "app" / "data" / "nc_uploads"


def allowed_nc_suffixes_text() -> str:
    return ", ".join(sorted(ALLOWED_NC_SUFFIXES))


def save_uploaded_nc_bytes(filename: str, raw: bytes) -> tuple[Path, str]:
    """写入 `app/data/nc_uploads/`，供 Streamlit / FastAPI 共用；返回 (绝对路径, task_id)。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_NC_SUFFIXES:
        raise ValueError(f"不支持的文件格式: {suffix}，允许 {allowed_nc_suffixes_text()}")
    size_mb = len(raw) / (1024**2)
    if size_mb > MAX_NC_UPLOAD_MB:
        raise ValueError(f"文件过大: {size_mb:.1f}MB，建议不超过 {MAX_NC_UPLOAD_MB}MB")
    task_id = uuid.uuid4().hex[:12]
    digest = hashlib.sha1(raw).hexdigest()[:8]
    safe = f"{task_id}_{digest}{suffix}"
    NC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = NC_CACHE_DIR / safe
    path.write_bytes(raw)
    return path, task_id


def save_uploaded_nc(uploaded_file: Any) -> tuple[Path, str]:
    """写入 `app/data/nc_uploads/`，返回 (绝对路径, task_id)。"""
    return save_uploaded_nc_bytes(str(uploaded_file.name), uploaded_file.getvalue())


def summarize_nc_bytes(raw: bytes, *, filename_hint: str | None = None) -> dict[str, Any]:
    """
    在系统临时目录验证上传字节是否为可读 NetCDF。
    避免先写入 `app/data/nc_uploads/` 触发 `uvicorn --reload` 整仓监视而导致
    「刚落盘文件即被重载打断 / 打开时报 FileNotFoundError」的竞态。
    """
    hint = Path(filename_hint or "upload.nc")
    ext = hint.suffix.lower()
    if ext not in ALLOWED_NC_SUFFIXES:
        ext = ".nc"
    fd, name = tempfile.mkstemp(suffix=ext)
    tmp = Path(name)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
    try:
        return summarize_nc_file(tmp)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def summarize_nc_file(path: str | Path) -> dict[str, Any]:
    """不加载全量数据，仅维度与变量名等摘要。"""
    p = Path(path)
    out: dict[str, Any] = {"path": str(p.resolve()), "exists": p.is_file(), "error": None}
    if not p.is_file():
        out["error"] = "文件不存在"
        return out
    try:
        import netCDF4 as nc  # type: ignore

        ds = nc.Dataset(str(p), mode="r")
        try:
            dims = {str(k): int(len(ds.dimensions[k])) for k in ds.dimensions}
            var_names = [str(k) for k in ds.variables.keys()]
            out["dimensions"] = dims
            out["variables"] = var_names
            out["size_mb"] = round(p.stat().st_size / (1024**2), 4)
        finally:
            ds.close()
    except Exception as e:
        out["error"] = str(e)
    return out


def cleanup_old_nc_uploads(max_files: int = 30) -> None:
    if not NC_CACHE_DIR.is_dir():
        return
    files = [x for x in NC_CACHE_DIR.iterdir() if x.is_file()]
    if len(files) <= max_files:
        return
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    for fp in files[max_files:]:
        try:
            fp.unlink()
        except OSError:
            pass
