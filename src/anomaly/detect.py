"""异常检测与台风检索（推理）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from src.anomaly.typhoon_kb import QueryBox, query_typhoon_events


def _parse_time(text: str) -> datetime:
    raw = str(text).strip()
    fmts = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S")
    for fmt in fmts:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {text}")


def _first_non_empty(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in mapping and mapping[k] not in (None, ""):
            return mapping[k]
    return None


def _build_query_from_anomaly_result(anomaly_result: dict[str, Any]) -> QueryBox:
    """
    从异常检测结果中提取时间窗与区域，兼容常见字段名：
    - 时间：start_time/end_time 或 time_start/time_end
    - 空间：lon_min/lon_max/lat_min/lat_max 或 bbox=[lon_min, lon_max, lat_min, lat_max]
    """
    start_text = _first_non_empty(anomaly_result, ("start_time", "time_start"))
    end_text = _first_non_empty(anomaly_result, ("end_time", "time_end"))
    if start_text is None or end_text is None:
        raise KeyError("异常结果缺少时间窗字段，需包含 start_time/end_time（或 time_start/time_end）")

    lon_min = _first_non_empty(anomaly_result, ("lon_min",))
    lon_max = _first_non_empty(anomaly_result, ("lon_max",))
    lat_min = _first_non_empty(anomaly_result, ("lat_min",))
    lat_max = _first_non_empty(anomaly_result, ("lat_max",))

    if any(v is None for v in (lon_min, lon_max, lat_min, lat_max)):
        bbox = anomaly_result.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            lon_min, lon_max, lat_min, lat_max = bbox

    if any(v is None for v in (lon_min, lon_max, lat_min, lat_max)):
        raise KeyError("异常结果缺少区域字段，需包含 lon/lat min/max 或 bbox=[lon_min, lon_max, lat_min, lat_max]")

    return QueryBox(
        start_time=_parse_time(str(start_text)),
        end_time=_parse_time(str(end_text)),
        lon_min=float(lon_min),
        lon_max=float(lon_max),
        lat_min=float(lat_min),
        lat_max=float(lat_max),
    )


def _zscore(value: float, mean: float, std: float) -> float:
    s = max(float(std), 1e-6)
    return (float(value) - float(mean)) / s


def compute_anomaly_assessment(anomaly_result: dict[str, Any]) -> dict[str, Any]:
    """根据风速/浪高残差输出 3sigma 异常指数与等级。"""
    wind_residual = anomaly_result.get("wind_residual")
    wave_residual = anomaly_result.get("wave_residual")
    has_signal = (wind_residual is not None) or (wave_residual is not None)
    if wind_residual is None:
        observed = anomaly_result.get("wind_observed")
        predicted = anomaly_result.get("wind_predicted")
        if observed is not None and predicted is not None:
            wind_residual = float(observed) - float(predicted)
            has_signal = True
    if wave_residual is None:
        observed = anomaly_result.get("wave_observed")
        predicted = anomaly_result.get("wave_predicted")
        if observed is not None and predicted is not None:
            wave_residual = float(observed) - float(predicted)
            has_signal = True

    if not has_signal:
        # 视频/涡旋演示路径：无水文残差时，用 peak_score 与时序曲线构造代理量（非实况观测量）
        peak_raw = anomaly_result.get("peak_score")
        curve = anomaly_result.get("current_curve")
        curve_list: list[float] = []
        if isinstance(curve, list):
            curve_list = [float(x) for x in curve if isinstance(x, (int, float))]
        peak_v: float | None = float(peak_raw) if isinstance(peak_raw, (int, float)) else None
        curve_std = float(np.std(np.asarray(curve_list, dtype=np.float64))) if curve_list else 0.0
        if peak_v is not None or curve_std > 0.0:
            wr = float(peak_v) if peak_v is not None else 0.0
            vr = float(curve_std) if curve_std > 0.0 else wr * 0.35
            wm = float(anomaly_result.get("wind_mean", 0.0))
            ws = max(float(anomaly_result.get("wind_std", 0.22)), 1e-6)
            vm = float(anomaly_result.get("wave_mean", 0.0))
            vs = max(float(anomaly_result.get("wave_std", 0.12)), 1e-6)
            wind_z = abs(_zscore(wr, wm, ws))
            wave_z = abs(_zscore(vr, vm, vs))
            anomaly_index = 0.5 * wind_z + 0.5 * wave_z
            if anomaly_index >= 3.0:
                anomaly_level = "high"
            elif anomaly_index >= 2.0:
                anomaly_level = "medium"
            else:
                anomaly_level = "low"
            return {
                "wind_residual": wr,
                "wave_residual": vr,
                "wind_z": float(wind_z),
                "wave_z": float(wave_z),
                "anomaly_index": float(anomaly_index),
                "anomaly_level": anomaly_level,
                "threshold_rule": "3sigma",
                "assessment_note": "演示代理：由 peak_score/时序分数推导，非水文观测残差；正式评测需接入命题方风浪要素。",
            }
        return {
            "wind_residual": 0.0,
            "wave_residual": 0.0,
            "wind_z": 0.0,
            "wave_z": 0.0,
            "anomaly_index": None,
            "anomaly_level": "unknown",
            "threshold_rule": "3sigma",
            "assessment_note": "缺少残差或观测/预测字段，无法完成异常分级",
        }

    wr = float(wind_residual) if wind_residual is not None else 0.0
    vr = float(wave_residual) if wave_residual is not None else 0.0
    wind_z = abs(_zscore(wr, anomaly_result.get("wind_mean", 0.0), anomaly_result.get("wind_std", 1.0)))
    wave_z = abs(_zscore(vr, anomaly_result.get("wave_mean", 0.0), anomaly_result.get("wave_std", 1.0)))
    anomaly_index = 0.5 * wind_z + 0.5 * wave_z
    if anomaly_index >= 3.0:
        anomaly_level = "high"
    elif anomaly_index >= 2.0:
        anomaly_level = "medium"
    else:
        anomaly_level = "low"

    out = {
        "wind_residual": float(wr),
        "wave_residual": float(vr),
        "wind_z": float(wind_z),
        "wave_z": float(wave_z),
        "anomaly_index": float(anomaly_index),
        "anomaly_level": anomaly_level,
        "threshold_rule": "3sigma",
    }
    note = anomaly_result.get("assessment_note")
    if isinstance(note, str) and note.strip():
        out["assessment_note"] = note.strip()
    return out


def _dtw_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("inf")
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, na + 1):
        ai = float(a[i - 1])
        for j in range(1, nb + 1):
            bj = float(b[j - 1])
            cost = abs(ai - bj)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb])


def _extract_curve_from_candidate(candidate: dict[str, Any], target_len: int) -> list[float]:
    for key in ("sequence", "curve", "wind_wave_curve"):
        raw = candidate.get(key)
        if isinstance(raw, list) and raw:
            return [float(v) for v in raw]
    for scalar_key in ("max_wind_kt", "max_wind", "peak_value"):
        if scalar_key in candidate and candidate.get(scalar_key) is not None:
            v = float(candidate[scalar_key])
            return [v for _ in range(max(target_len, 1))]
    return []


def rerank_candidates_by_dtw(
    *,
    candidates: list[dict[str, Any]],
    current_curve: list[float] | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not current_curve:
        return candidates[:top_k], {"enabled": False, "reason": "missing_current_curve"}

    enriched: list[dict[str, Any]] = []
    for item in candidates:
        cand_curve = _extract_curve_from_candidate(item, len(current_curve))
        dist = _dtw_distance(current_curve, cand_curve) if cand_curve else float("inf")
        row = dict(item)
        row["dtw_distance"] = float(dist)
        enriched.append(row)
    enriched.sort(key=lambda x: float(x.get("dtw_distance", float("inf"))))
    return enriched[:top_k], {"enabled": True, "distance_field": "dtw_distance"}


def link_anomaly_to_typhoon(
    *,
    anomaly_result: dict[str, Any],
    events_json_path: str = "data/processed/anomaly/typhoon_kb/events.json",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    将异常检测结果联动到台风知识库检索并用 DTW 重排（若可用）。

    返回结构包含：
    - query: 标准化后的时间窗与空间范围
    - candidates: 候选台风事件列表
    - linked: 是否命中候选
    """
    query = _build_query_from_anomaly_result(anomaly_result)
    candidates = query_typhoon_events(
        events_json_path=events_json_path,
        query=query,
        top_k=max(int(top_k), 1) * 3,
    )
    current_curve = anomaly_result.get("current_curve")
    if isinstance(current_curve, list):
        current = [float(v) for v in current_curve]
    else:
        current = None
    reranked, dtw_meta = rerank_candidates_by_dtw(candidates=candidates, current_curve=current, top_k=int(top_k))

    return {
        "query": {
            "start_time": query.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": query.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "lon_min": query.lon_min,
            "lon_max": query.lon_max,
            "lat_min": query.lat_min,
            "lat_max": query.lat_max,
        },
        "events_json_path": events_json_path,
        "top_k": int(top_k),
        "linked": bool(reranked),
        "candidates": reranked,
        "retrieval": {
            "method": "time_space_filter+dtw_rerank",
            "dtw": dtw_meta,
        },
    }


def run_detect(
    *,
    anomaly_result: dict[str, Any],
    auto_link_typhoon: bool = True,
    events_json_path: str = "data/processed/anomaly/typhoon_kb/events.json",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    模块 C 推理主链入口：
    - 输出异常等级（3sigma）
    - 自动台风检索（时间窗+区域）并在可用时进行 DTW 重排
    """
    assessment = compute_anomaly_assessment(anomaly_result)
    merged_result = dict(anomaly_result)
    merged_result.update(assessment)
    out = {"anomaly_result": merged_result}
    if not auto_link_typhoon:
        out["typhoon_link"] = {"linked": False, "reason": "auto_link_typhoon=false"}
        return out
    out["typhoon_link"] = link_anomaly_to_typhoon(
        anomaly_result=merged_result,
        events_json_path=events_json_path,
        top_k=top_k,
    )
    return out
