"""异常检测与台风检索（推理）。

模块结构：
  1. 通用工具
  2. 异常分级（3σ）：逐时次分段 + 全局 assessment
  3. 异常事件窗：主窗提取与序列切片
  4. DTW 查询曲线：构造与解析
  5. DTW 重排：距离计算与候选排序
  6. 台风检索联动
  7. 公共入口 run_detect
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from src.anomaly.typhoon_kb import KT_TO_MPS, QueryBox, query_typhoon_events


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


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


def _zscore(value: float, mean: float, std: float) -> float:
    s = max(float(std), 1e-6)
    return (float(value) - float(mean)) / s


# ---------------------------------------------------------------------------
# 异常分级（3σ）
# ---------------------------------------------------------------------------


def _level_from_index(anomaly_index: float) -> str:
    if anomaly_index >= 3.0:
        return "high"
    if anomaly_index >= 2.0:
        return "medium"
    if anomaly_index >= 1.0:
        return "low"
    return "normal"


def compute_series_anomaly_segments(
    *,
    wind_observed: list[float],
    wind_predicted: list[float],
    wave_observed: list[float],
    wave_predicted: list[float],
) -> list[dict[str, Any]]:
    """
    逐时次 3σ 异常分段（§5.4.4）：黄/橙/红对应 low/medium/high。
    返回每步 anomaly_index 与 level，供前端曲线高亮。
    """
    n = min(len(wind_observed), len(wind_predicted), len(wave_observed), len(wave_predicted))
    if n < 1:
        return []
    wo = np.asarray(wind_observed[:n], dtype=np.float64)
    wp = np.asarray(wind_predicted[:n], dtype=np.float64)
    ho = np.asarray(wave_observed[:n], dtype=np.float64)
    hp = np.asarray(wave_predicted[:n], dtype=np.float64)
    wd = wo - wp
    hd = ho - hp
    ws = max(float(np.std(wd)), 0.05)
    vs = max(float(np.std(hd)), 0.02)
    out: list[dict[str, Any]] = []
    for i in range(n):
        wz = abs(_zscore(float(wd[i]), 0.0, ws))
        vz = abs(_zscore(float(hd[i]), 0.0, vs))
        idx = 0.5 * wz + 0.5 * vz
        out.append(
            {
                "step": i,
                "anomaly_index": round(float(idx), 4),
                "level": _level_from_index(float(idx)),
                "wind_residual": round(float(wd[i]), 4),
                "wave_residual": round(float(hd[i]), 4),
            }
        )
    return out


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


# ---------------------------------------------------------------------------
# 异常事件窗：主窗提取与序列切片
# ---------------------------------------------------------------------------


def slice_series_by_window(series: list[float], t0: int, t1: int) -> list[float]:
    """闭区间 [t0, t1] 切片（含端点）。"""
    if not series:
        return []
    lo = max(0, int(t0))
    hi = min(len(series) - 1, int(t1))
    if hi < lo:
        return []
    return [float(v) for v in series[lo : hi + 1]]


def extract_primary_anomaly_window(
    segments: list[dict[str, Any]],
    *,
    tau: float = 1.5,
    min_len: int = 2,
    gap_merge: int = 1,
    pad: int = 4,
    peak_half_width: int = 5,
    n_steps: int | None = None,
) -> dict[str, Any]:
    """
    从逐时次 anomaly_index 提取主异常事件窗；无连续超阈段时 peak_centered fallback。
    """
    n = int(n_steps if n_steps is not None else len(segments))
    if n < 1:
        return {
            "t_start": 0,
            "t_end": 0,
            "t_start_padded": 0,
            "t_end_padded": 0,
            "fallback_reason": "empty_series",
            "tau": float(tau),
        }

    indices = [float(s.get("anomaly_index", 0.0)) for s in segments[:n]]
    while len(indices) < n:
        indices.append(0.0)

    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if indices[i] >= tau:
            j = i + 1
            while j < n and indices[j] >= tau:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if runs:
        merged: list[tuple[int, int]] = [runs[0]]
        for start, end in runs[1:]:
            ps, pe = merged[-1]
            if start - pe - 1 <= gap_merge:
                merged[-1] = (ps, end)
            else:
                merged.append((start, end))
        t_start, t_end = max(merged, key=lambda r: max(indices[r[0] : r[1] + 1]))
        fallback_reason: str | None = None
    else:
        peak_i = int(np.argmax(np.asarray(indices, dtype=np.float64)))
        t_start = max(0, peak_i - peak_half_width)
        t_end = min(n - 1, peak_i + peak_half_width)
        fallback_reason = "peak_centered"

    t_start_p = max(0, t_start - pad)
    t_end_p = min(n - 1, t_end + pad)
    return {
        "t_start": int(t_start),
        "t_end": int(t_end),
        "t_start_padded": int(t_start_p),
        "t_end_padded": int(t_end_p),
        "fallback_reason": fallback_reason,
        "tau": float(tau),
    }


# ---------------------------------------------------------------------------
# DTW 查询曲线：构造与解析
# ---------------------------------------------------------------------------


def build_wind_dtw_query_curve(
    *,
    wind_observed: list[float],
    wind_predicted: list[float] | None = None,
    wave_observed: list[float] | None = None,
    wave_predicted: list[float] | None = None,
    segments: list[dict[str, Any]] | None = None,
    mode: str = "regional_mean_obs_vs_ibtracs_center",
    dtw_config: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """
    构造 DTW 查询曲线与元数据。
    regional_mean_obs_vs_ibtracs_center：异常窗内区域平均 wind_observed。
    wind_residual_vs_ibtracs_track（legacy）：全长 |obs−pred|。
    """
    cfg = dtw_config or {}
    meta: dict[str, Any] = {"match_mode": mode}
    n = len(wind_observed)
    if n < 1:
        meta["reason"] = "missing_wind_observed"
        return [], meta

    if mode == "wind_residual_vs_ibtracs_track":
        if wind_predicted is None or len(wind_predicted) != n:
            meta["reason"] = "missing_wind_predicted"
            return [], meta
        woa = np.asarray(wind_observed, dtype=np.float64)
        wpa = np.asarray(wind_predicted, dtype=np.float64)
        curve = np.abs(woa - wpa).astype(float).tolist()
        meta["query_curve"] = "wind_residual_full"
        meta["window"] = {"t_start_padded": 0, "t_end_padded": n - 1, "fallback_reason": "legacy_full_residual"}
        return curve, meta

    if segments is None and wind_predicted is not None and wave_observed is not None and wave_predicted is not None:
        segments = compute_series_anomaly_segments(
            wind_observed=wind_observed,
            wind_predicted=wind_predicted,
            wave_observed=wave_observed,
            wave_predicted=wave_predicted,
        )
    if not segments:
        meta["reason"] = "missing_segments"
        return [], meta

    window = extract_primary_anomaly_window(
        segments,
        tau=float(cfg.get("dtw_window_tau", 1.5)),
        min_len=int(cfg.get("dtw_window_min_len", 2)),
        gap_merge=int(cfg.get("dtw_window_gap_merge", 1)),
        pad=int(cfg.get("dtw_window_pad", 4)),
        peak_half_width=int(cfg.get("dtw_window_fallback_peak_half_width", 5)),
        n_steps=n,
    )
    curve = slice_series_by_window(wind_observed, window["t_start_padded"], window["t_end_padded"])
    meta["query_curve"] = "wind_obs_regional_mean_window"
    meta["window"] = window
    meta["fallback_reason"] = window.get("fallback_reason")
    return curve, meta


def _resolve_wind_dtw_query_curve(anomaly_result: dict[str, Any]) -> tuple[list[float] | None, dict[str, Any]]:
    """DTW 查询侧：优先已组装的 wind_dtw_curve；否则按 dtw_match_mode 现算。"""
    meta: dict[str, Any] = {}
    raw = anomaly_result.get("wind_dtw_curve")
    if isinstance(raw, list) and raw:
        meta["match_mode"] = anomaly_result.get("dtw_match_mode")
        if isinstance(anomaly_result.get("anomaly_event_window"), dict):
            meta["window"] = anomaly_result["anomaly_event_window"]
        meta["fallback_reason"] = anomaly_result.get("dtw_fallback_reason")
        meta["query_curve"] = anomaly_result.get(
            "dtw_query_curve",
            "wind_obs_regional_mean_window"
            if anomaly_result.get("dtw_match_mode") == "regional_mean_obs_vs_ibtracs_center"
            else "wind_dtw_curve",
        )
        return [float(v) for v in raw], meta

    from src.anomaly.dtw_config import DEFAULT_DTW_MATCH_MODE, load_dtw_link_config

    cfg = load_dtw_link_config()
    mode = str(anomaly_result.get("dtw_match_mode") or cfg.get("dtw_match_mode") or DEFAULT_DTW_MATCH_MODE)
    wo = _first_non_empty(anomaly_result, ("wind_observed", "demo_wind_observed"))
    wp = _first_non_empty(anomaly_result, ("wind_predicted", "demo_wind_predicted"))
    ho = _first_non_empty(anomaly_result, ("wave_observed", "demo_wave_observed"))
    hp = _first_non_empty(anomaly_result, ("wave_predicted", "demo_wave_predicted"))
    if not isinstance(wo, list) or not wo:
        return None, {"reason": "missing_wind_observed", "match_mode": mode}

    segments = anomaly_result.get("anomaly_segments")
    if not isinstance(segments, list):
        segments = None
    wp_list = wp if isinstance(wp, list) else None
    ho_list = ho if isinstance(ho, list) else None
    hp_list = hp if isinstance(hp, list) else None
    curve, build_meta = build_wind_dtw_query_curve(
        wind_observed=[float(v) for v in wo],
        wind_predicted=[float(v) for v in wp_list] if wp_list else None,
        wave_observed=[float(v) for v in ho_list] if ho_list else None,
        wave_predicted=[float(v) for v in hp_list] if hp_list else None,
        segments=segments,
        mode=mode,
        dtw_config=cfg,
    )
    if not curve:
        return None, build_meta
    return curve, build_meta


# ---------------------------------------------------------------------------
# DTW 重排：距离计算与候选排序
# ---------------------------------------------------------------------------


def _znorm_curve(curve: list[float]) -> list[float]:
    """DTW 弱匹配：对曲线 z-score，比较时间演化形态而非绝对量纲。"""
    a = np.asarray(curve, dtype=np.float64)
    if a.size == 0:
        return []
    std = max(float(np.std(a)), 1e-6)
    return ((a - float(np.mean(a))) / std).astype(float).tolist()


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
    track_mps = candidate.get("wind_track_mps")
    if isinstance(track_mps, list) and track_mps:
        return [float(v) for v in track_mps]
    track_kt = candidate.get("wind_track_kt")
    if isinstance(track_kt, list) and track_kt:
        return [float(v) * KT_TO_MPS for v in track_kt]
    for key in ("sequence", "curve", "wind_wave_curve"):
        raw = candidate.get(key)
        if isinstance(raw, list) and raw:
            return [float(v) for v in raw]
    for scalar_key in ("peak_wind_kt", "max_wind_kt", "max_wind", "peak_value"):
        if scalar_key in candidate and candidate.get(scalar_key) is not None:
            v = float(candidate[scalar_key])
            if scalar_key == "peak_wind_kt" or scalar_key == "max_wind_kt":
                v = v * KT_TO_MPS
            return [v for _ in range(max(target_len, 1))]
    return []


def rerank_candidates_by_dtw(
    *,
    candidates: list[dict[str, Any]],
    current_curve: list[float] | None,
    top_k: int,
    normalize: bool = True,
    match_mode: str = "regional_mean_obs_vs_ibtracs_center",
    query_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not current_curve:
        out_meta: dict[str, Any] = {
            "enabled": False,
            "reason": "missing_wind_dtw_curve",
            "match_mode": match_mode,
        }
        if query_meta:
            out_meta.update({k: v for k, v in query_meta.items() if k not in out_meta})
        return candidates[:top_k], out_meta

    query = _znorm_curve(current_curve) if normalize else [float(v) for v in current_curve]

    enriched: list[dict[str, Any]] = []
    used_track = 0
    used_peak_fallback = 0
    for item in candidates:
        cand_curve = _extract_curve_from_candidate(item, len(current_curve))
        if not cand_curve:
            dist = float("inf")
        else:
            if item.get("wind_track_mps") or item.get("wind_track_kt"):
                used_track += 1
            elif item.get("peak_wind_kt") is not None:
                used_peak_fallback += 1
            cand = _znorm_curve(cand_curve) if normalize else cand_curve
            dist = _dtw_distance(query, cand)
        row = dict(item)
        row["dtw_distance"] = float(dist)
        enriched.append(row)
    enriched.sort(key=lambda x: float(x.get("dtw_distance", float("inf"))))
    qm = query_meta or {}
    query_curve = qm.get("query_curve")
    if not query_curve:
        query_curve = (
            "wind_obs_regional_mean_window"
            if match_mode == "regional_mean_obs_vs_ibtracs_center"
            else "wind_dtw_curve"
        )
    dtw_meta: dict[str, Any] = {
        "enabled": True,
        "distance_field": "dtw_distance",
        "match_mode": match_mode,
        "query_curve": query_curve,
        "history_curve": "wind_track_mps",
        "normalized": bool(normalize),
        "n_candidates_with_track": used_track,
        "n_candidates_peak_fallback": used_peak_fallback,
    }
    for key in ("window", "fallback_reason"):
        if key in qm and qm[key] is not None:
            dtw_meta[key] = qm[key]
    return enriched[:top_k], dtw_meta


# ---------------------------------------------------------------------------
# 台风检索联动
# ---------------------------------------------------------------------------


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


def link_anomaly_to_typhoon(
    *,
    anomaly_result: dict[str, Any],
    events_json_path: str = "data/processed/anomaly/typhoon_kb/events.json",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    将异常检测结果联动到台风查询检索并用 DTW 重排（若可用）。

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
    from src.anomaly.dtw_config import DEFAULT_DTW_MATCH_MODE, load_dtw_link_config

    cfg = load_dtw_link_config()
    match_mode = str(anomaly_result.get("dtw_match_mode") or cfg.get("dtw_match_mode") or DEFAULT_DTW_MATCH_MODE)
    wind_query, query_build_meta = _resolve_wind_dtw_query_curve(anomaly_result)
    reranked, dtw_meta = rerank_candidates_by_dtw(
        candidates=candidates,
        current_curve=wind_query,
        top_k=int(top_k),
        match_mode=match_mode,
        query_meta=query_build_meta,
    )

    query_out: dict[str, Any] = {
        "start_time": query.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": query.end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "lon_min": query.lon_min,
        "lon_max": query.lon_max,
        "lat_min": query.lat_min,
        "lat_max": query.lat_max,
    }
    for meta_key in (
        "anomaly_start_time",
        "anomaly_end_time",
        "nc_coverage_start_time",
        "nc_coverage_end_time",
        "history_search_mode",
        "history_lookback_years",
    ):
        if meta_key in anomaly_result:
            query_out[meta_key] = anomaly_result[meta_key]

    return {
        "query": query_out,
        "events_json_path": events_json_path,
        "top_k": int(top_k),
        "linked": bool(reranked),
        "candidates": reranked,
        "retrieval": {
            "method": "time_space_filter+dtw_wind_process_rerank",
            "dtw": dtw_meta,
        },
    }


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


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
