"""Tests for regional-mean obs window DTW query construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.anomaly.detect import (
    build_wind_dtw_query_curve,
    compute_series_anomaly_segments,
    extract_primary_anomaly_window,
    link_anomaly_to_typhoon,
    slice_series_by_window,
)


def test_slice_series_by_window_inclusive() -> None:
    s = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert slice_series_by_window(s, 1, 3) == [2.0, 3.0, 4.0]


def test_extract_primary_anomaly_window_finds_run() -> None:
    segments = [{"anomaly_index": 0.4} for _ in range(10)]
    segments[4]["anomaly_index"] = 2.0
    segments[5]["anomaly_index"] = 2.2
    w = extract_primary_anomaly_window(segments, tau=1.5, min_len=2, pad=2)
    assert w["t_start"] == 4
    assert w["t_end"] == 5
    assert w["t_start_padded"] == 2
    assert w["fallback_reason"] is None


def test_extract_primary_anomaly_window_peak_fallback() -> None:
    segments = [{"anomaly_index": 0.2} for _ in range(8)]
    segments[3]["anomaly_index"] = 1.1
    w = extract_primary_anomaly_window(segments, tau=1.5, min_len=2, peak_half_width=2, pad=1)
    assert w["fallback_reason"] == "peak_centered"
    assert w["t_start"] <= 3 <= w["t_end"]


def test_build_wind_dtw_regional_mean_window_shorter_than_full() -> None:
    n = 16
    wo = [5.0 + 0.1 * i for i in range(n)]
    wp = wo[:]
    ho = [1.0] * n
    hp = [1.0] * n
    segments = compute_series_anomaly_segments(
        wind_observed=wo,
        wind_predicted=wp,
        wave_observed=ho,
        wave_predicted=hp,
    )
    for i in (8, 9, 10):
        segments[i]["anomaly_index"] = 2.5
    curve, meta = build_wind_dtw_query_curve(
        wind_observed=wo,
        wind_predicted=wp,
        wave_observed=ho,
        wave_predicted=hp,
        segments=segments,
        mode="regional_mean_obs_vs_ibtracs_center",
        dtw_config={"dtw_window_tau": 1.5, "dtw_window_min_len": 2, "dtw_window_pad": 2},
    )
    assert meta["query_curve"] == "wind_obs_regional_mean_window"
    assert 0 < len(curve) < n


def test_build_wind_dtw_legacy_residual_full_length() -> None:
    wo = [1.0, 2.0, 3.0]
    wp = [1.1, 2.2, 2.5]
    curve, meta = build_wind_dtw_query_curve(
        wind_observed=wo,
        wind_predicted=wp,
        wave_observed=[0.5, 0.5, 0.5],
        wave_predicted=[0.5, 0.5, 0.5],
        mode="wind_residual_vs_ibtracs_track",
    )
    assert len(curve) == 3
    assert meta["query_curve"] == "wind_residual_full"
    assert curve[1] == pytest.approx(0.2, abs=1e-6)


def test_tau_15_vs_20_window_coverage() -> None:
    segments = [{"anomaly_index": 0.3} for _ in range(16)]
    segments[6]["anomaly_index"] = 1.6
    segments[7]["anomaly_index"] = 1.6
    w15 = extract_primary_anomaly_window(segments, tau=1.5, min_len=2)
    w20 = extract_primary_anomaly_window(segments, tau=2.0, min_len=2)
    assert w15["fallback_reason"] is None
    assert w20["fallback_reason"] == "peak_centered"


@pytest.mark.skipif(
    not Path("data/processed/anomaly/typhoon_kb/events.json").is_file(),
    reason="typhoon kb missing",
)
def test_link_anomaly_dtw_meta_match_mode() -> None:
    n = 12
    wo = [10.0 + 0.2 * i for i in range(n)]
    wp = [9.8 + 0.2 * i for i in range(n)]
    segments = compute_series_anomaly_segments(
        wind_observed=wo,
        wind_predicted=wp,
        wave_observed=[1.0] * n,
        wave_predicted=[1.0] * n,
    )
    curve, _ = build_wind_dtw_query_curve(
        wind_observed=wo,
        wind_predicted=wp,
        wave_observed=[1.0] * n,
        wave_predicted=[1.0] * n,
        segments=segments,
        mode="regional_mean_obs_vs_ibtracs_center",
    )
    ar = {
        "start_time": "2020-01-01 00:00:00",
        "end_time": "2024-12-31 23:59:59",
        "lon_min": 117.0,
        "lon_max": 127.0,
        "lat_min": 31.0,
        "lat_max": 41.0,
        "wind_dtw_curve": curve,
        "dtw_match_mode": "regional_mean_obs_vs_ibtracs_center",
        "dtw_query_curve": "wind_obs_regional_mean_window",
        "anomaly_event_window": {"t_start_padded": 0, "t_end_padded": len(curve) - 1},
        "wind_observed": wo,
        "wind_predicted": wp,
        "wave_observed": [1.0] * n,
        "wave_predicted": [1.0] * n,
        "anomaly_segments": segments,
    }
    link = link_anomaly_to_typhoon(anomaly_result=ar, top_k=3)
    dtw = (link.get("retrieval") or {}).get("dtw") or {}
    assert dtw.get("match_mode") == "regional_mean_obs_vs_ibtracs_center"
    assert dtw.get("query_curve") == "wind_obs_regional_mean_window"
