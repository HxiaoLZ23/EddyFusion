from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.preprocess.nc_lazy_subset import load_nc_variable_map, probe_nc_meta, subset_netcdf
from web_api.deps import REPO_ROOT, resolve_nc_token

router = APIRouter(prefix="/preprocess", tags=["preprocess"])


@router.get("/variable-map")
def get_variable_map() -> dict[str, Any]:
    """论文表 5-2 变量映射（只读）。"""
    return load_nc_variable_map()


@router.get("/meta")
def get_nc_meta(nc_path: str = Query(..., description="白名单 NC 相对路径")) -> dict[str, Any]:
    p = resolve_nc_token(nc_path)
    try:
        meta = probe_nc_meta(p)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"NC 元数据探测失败: {e}") from e
    rel = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    meta["nc_path"] = rel
    return meta


class SubsetBody(BaseModel):
    nc_path: str = Field(..., description="源 NC 相对路径")
    time_start: int | None = Field(None, ge=0, description="时间维起始索引（含）")
    time_stop: int | None = Field(None, ge=0, description="时间维结束索引（含）")
    lon_min: float | None = None
    lon_max: float | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    task: Literal["eddy", "windwave"] | None = Field(None, description="任务类型校验")


@router.post("/subset")
def post_subset(body: SubsetBody) -> dict[str, Any]:
    p = resolve_nc_token(body.nc_path)
    try:
        out = subset_netcdf(
            p,
            time_start=body.time_start,
            time_stop=body.time_stop,
            lon_min=body.lon_min,
            lon_max=body.lon_max,
            lat_min=body.lat_min,
            lat_max=body.lat_max,
            task=body.task,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"裁剪失败: {e}") from e
    return out
