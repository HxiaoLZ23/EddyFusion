from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.nc_ingest_service import save_uploaded_nc_bytes, summarize_nc_bytes
from web_api.deps import REPO_ROOT

router = APIRouter(prefix="/offline", tags=["offline"])


@router.post("/nc")
async def upload_nc(files: list[UploadFile] = File(...)) -> dict[str, list[str]]:
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件（字段名 files）")
    paths: list[str] = []
    for f in files:
        raw = await f.read()
        summ = summarize_nc_bytes(raw, filename_hint=f.filename or "upload.nc")
        if summ.get("error"):
            raise HTTPException(
                status_code=400,
                detail=f"文件「{f.filename}」不是可读 NetCDF（或已损坏）: {summ['error']}",
            )
        try:
            abs_path, _tid = save_uploaded_nc_bytes(f.filename or "upload.nc", raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        rel = abs_path.resolve().relative_to(REPO_ROOT.resolve())
        paths.append(rel.as_posix())
    return {"paths": paths}
