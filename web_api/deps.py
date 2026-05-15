from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

from services.nc_ingest_service import ALLOWED_NC_SUFFIXES, NC_CACHE_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
HYDRO_NC_CACHE = REPO_ROOT / "app" / "data" / "hydro_nc_cache"
EDDY_PREVIEW_DIR = REPO_ROOT / "app" / "data" / "eddy_preview"


def allowed_nc_roots() -> tuple[Path, ...]:
    extra = os.environ.get("REALTIME_NC_POLL_DIR", "").strip()
    roots = [NC_CACHE_DIR.resolve(), HYDRO_NC_CACHE.resolve()]
    if extra:
        roots.append(Path(extra).expanduser().resolve())
    return tuple(roots)


def resolve_nc_token(token: str) -> Path:
    """
    允许：仓库根相对路径，或已解析且落在 nc 上传 / hydro 缓存 / REALTIME_NC_POLL_DIR 下的绝对路径。
    """
    raw = (token or "").strip().strip('"').strip("'").replace("\\", "/")
    if not raw:
        raise HTTPException(status_code=400, detail="非法路径")
    trial = Path(raw)
    if trial.is_absolute():
        p = trial.resolve()
    else:
        if ".." in Path(raw).parts:
            raise HTTPException(status_code=400, detail="非法路径")
        p = (REPO_ROOT / raw).resolve()
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {raw}")
    allowed = allowed_nc_roots()
    ok = False
    for r in allowed:
        rp = r.resolve()
        try:
            p.relative_to(rp)
            ok = True
            break
        except ValueError:
            continue
    if not ok:
        raise HTTPException(status_code=400, detail="路径不在允许的 NC 根目录下")
    if p.suffix.lower() not in ALLOWED_NC_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 NetCDF 后缀")
    return p


def resolve_repo_relative(rel: str, *, must_exist: bool = True) -> Path:
    raw = (rel or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or ".." in Path(raw).parts:
        raise HTTPException(status_code=400, detail="非法配置路径")
    p = (REPO_ROOT / raw).resolve()
    if must_exist and not p.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {rel}")
    if not str(p).startswith(str(REPO_ROOT.resolve()) + os.sep) and p != REPO_ROOT:
        raise HTTPException(status_code=400, detail="路径必须位于仓库根目录下")
    return p


def resolve_eddy_preview_file(filename: str) -> Path:
    """仅允许 `app/data/eddy_preview/` 下的涡旋预览 MP4 文件名。"""
    name = (filename or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="非法文件名")
    if ".." in Path(name).parts or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    p = (EDDY_PREVIEW_DIR / name).resolve()
    root = EDDY_PREVIEW_DIR.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="路径逃逸") from None
    if not p.is_file():
        raise HTTPException(status_code=404, detail="预览文件不存在")
    return p
