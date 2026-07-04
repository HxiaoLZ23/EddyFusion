from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.hydro_inference_service import HydroInferenceService
from src.hydro.physical_scale import (
    denorm_array,
    feature_unit,
    feature_units_map,
    finite_vmin_vmax,
    resolve_zscore_stats,
)
from web_api.deps import resolve_nc_token, resolve_repo_relative
from web_api.grid import lonlat_from_hydro_nc, sorted_nc_paths

router = APIRouter(prefix="/hydro", tags=["hydro"])

_svc = HydroInferenceService()


def _nan_to_none_2d(arr: np.ndarray) -> list[list[float | None]]:
    out: list[list[float | None]] = []
    for row in np.asarray(arr):
        line: list[float | None] = []
        for v in np.asarray(row, dtype=float).ravel():
            if not math.isfinite(float(v)):
                line.append(None)
            else:
                line.append(float(v))
        out.append(line)
    return out


class HeatmapRequest(BaseModel):
    nc_paths: list[str] = Field(..., min_length=1)
    config_path: str = "config/hydro_hycom_l2.yaml"
    ckpt_path: str = "outputs/hydro_l2/best.pt"
    sample_index: int = 0
    window_stride: int = 24
    max_windows: int = 256
    feature: str = "temp"
    kind: str = "pred"
    lead_time_index: int = 0


class HydroMetaRequest(BaseModel):
    nc_paths: list[str] = Field(default_factory=list)
    config_path: str = "config/hydro_hycom_l2.yaml"


def _kind_to_key(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k in ("abs_err", "err", "|err|"):
        return "err"
    if k in ("gt", "pred"):
        return k
    raise HTTPException(status_code=400, detail="kind 须为 gt、pred 或 abs_err")


@router.post("/heatmap")
def hydro_heatmap(body: HeatmapRequest) -> dict[str, Any]:
    _ = resolve_repo_relative(body.config_path)
    _ = resolve_repo_relative(body.ckpt_path)
    nc_abs = [resolve_nc_token(p) for p in body.nc_paths]
    ordered = sorted_nc_paths(nc_abs)
    first = ordered[0]

    try:
        lons, lats = lonlat_from_hydro_nc(first, config_path=body.config_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"读取经纬度失败: {e}") from e

    try:
        xp, yp, meta = _svc.materialize_netcdf_to_xy_npz(
            ordered,
            config_path=body.config_path,
            window_stride=int(body.window_stride),
            max_windows=int(body.max_windows),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    map_key = _kind_to_key(body.kind)
    try:
        result = _svc.run(
            config_path=body.config_path,
            ckpt_path=body.ckpt_path,
            split="val",
            sample_index=int(body.sample_index),
            x_path_override=str(xp),
            y_path_override=str(yp),
            map_time_index=int(body.lead_time_index),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}") from e

    feat = body.feature
    if feat not in result.get("map_data", {}):
        raise HTTPException(
            status_code=400,
            detail=f"未知要素 {feat!r}，可用: {list(result.get('map_data', {}).keys())}",
        )
    maps = result["map_data"][feat]
    gt_map = np.asarray(maps["gt"], dtype=np.float64)
    pd_map = np.asarray(maps["pred"], dtype=np.float64)

    t_need = _svc.hydro_required_time_steps(body.config_path)
    t_hat = _svc.peek_hydro_buffer_time_steps([str(p) for p in ordered], config_path=body.config_path)

    feature_names = list(result.get("feature_names", []))
    warnings = list(result.get("warnings", []))
    mat_public = {k: v for k, v in meta.items() if k not in ("x_path", "y_path")}

    stats = resolve_zscore_stats(config_path=body.config_path, materialize_meta=meta)
    value_scale = "physical"
    value_unit = feature_unit(feat)
    if map_key == "err":
        value_unit = feature_unit(feat)  # |Δ| 与要素同量纲

    if stats is not None:
        mean_1d, std_1d, _ = stats
        if mat_public.get("stats_npz_rejected"):
            warnings = warnings + [str(mat_public["stats_npz_rejected"])]
        if feat not in feature_names:
            raise HTTPException(status_code=400, detail=f"要素 {feat!r} 不在配置 target_features 中")
        fi = feature_names.index(feat)
        gt_phys = denorm_array(gt_map, fi, mean_1d, std_1d)
        pd_phys = denorm_array(pd_map, fi, mean_1d, std_1d)
        if map_key == "gt":
            arr = gt_phys
        elif map_key == "pred":
            arr = pd_phys
        else:
            arr = np.abs(pd_phys - gt_phys)
    else:
        value_scale = "normalized"
        value_unit = "σ"
        warnings = warnings + ["未解析到 Z-score 统计量，热力图与曲线仍为标准化值。"]
        if map_key == "gt":
            arr = gt_map
        elif map_key == "pred":
            arr = pd_map
        else:
            arr = np.abs(pd_map - gt_map)

    vmin, vmax = finite_vmin_vmax(arr, floor_zero=(map_key == "err"))
    values = _nan_to_none_2d(arr)

    curve_raw = result.get("curve_data") or {}
    curve_json: dict[str, Any] = {}
    for name, rows in curve_raw.items():
        if not isinstance(rows, list):
            curve_json[str(name)] = []
            continue
        out_rows: list[dict[str, float]] = []
        if stats is not None and name in feature_names:
            j = feature_names.index(name)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                gt_z = float(row.get("gt", 0))
                pd_z = float(row.get("pred", 0))
                out_rows.append(
                    {
                        "horizon": float(row.get("horizon", 0)),
                        "gt": float(denorm_array(np.array([gt_z]), j, mean_1d, std_1d)[0]),
                        "pred": float(denorm_array(np.array([pd_z]), j, mean_1d, std_1d)[0]),
                    }
                )
        else:
            out_rows = [dict(row) for row in rows if isinstance(row, dict)]
        curve_json[str(name)] = out_rows

    return {
        "lons": lons,
        "lats": lats,
        "values": values,
        "feature": feat,
        "kind": body.kind,
        "lead_time_index": int(body.lead_time_index),
        "crs": "EPSG:4326",
        "value_scale": value_scale,
        "value_unit": value_unit,
        "vmin": vmin,
        "vmax": vmax,
        "feature_units": feature_units_map(feature_names),
        "warnings": warnings,
        "feature_names": feature_names,
        "curve_data": curve_json,
        "inference": {
            "nrmse_avg": result.get("nrmse_avg"),
            "sample_index": result.get("sample_index"),
            "elapsed_sec": result.get("elapsed_sec"),
            "t_last": result.get("t_last"),
        },
        "meta": {
            "T_need": t_need,
            "T_hat": t_hat,
            "buffer_sufficient": bool(t_hat >= t_need),
            "materialize": mat_public,
        },
    }


@router.post("/meta")
def hydro_meta(body: HydroMetaRequest) -> dict[str, Any]:
    _ = resolve_repo_relative(body.config_path)
    t_need = _svc.hydro_required_time_steps(body.config_path)
    if not body.nc_paths:
        return {"T_need": t_need, "T_hat": 0, "buffer_sufficient": False}
    nc_abs = [resolve_nc_token(p) for p in body.nc_paths]
    ordered = sorted_nc_paths(nc_abs)
    t_hat = _svc.peek_hydro_buffer_time_steps([str(p) for p in ordered], config_path=body.config_path)
    return {"T_need": t_need, "T_hat": t_hat, "buffer_sufficient": bool(t_hat >= t_need)}
