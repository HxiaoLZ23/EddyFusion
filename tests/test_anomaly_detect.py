from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.anomaly.detect import compute_anomaly_assessment, rerank_candidates_by_dtw, run_detect
from src.anomaly.eddy_typhoon_bridge import infer_typhoon_link_defaults_from_eddy_result
from src.anomaly.typhoon_kb import QueryBox, query_typhoon_events


def test_compute_anomaly_assessment_uses_three_sigma_levels() -> None:
    low = compute_anomaly_assessment(
        {
            "wind_residual": 0.4,
            "wave_residual": 0.2,
            "wind_mean": 0.0,
            "wave_mean": 0.0,
            "wind_std": 1.0,
            "wave_std": 1.0,
        }
    )
    high = compute_anomaly_assessment(
        {
            "wind_residual": 4.0,
            "wave_residual": 3.2,
            "wind_mean": 0.0,
            "wave_mean": 0.0,
            "wind_std": 1.0,
            "wave_std": 1.0,
        }
    )

    assert low["anomaly_level"] == "low"
    assert high["anomaly_level"] == "high"
    assert high["anomaly_index"] > low["anomaly_index"]


def test_rerank_candidates_by_dtw_prefers_closest_curve() -> None:
    candidates = [
        {"event_id": "far", "sequence": [10.0, 10.0, 10.0]},
        {"event_id": "near", "sequence": [0.0, 1.0, 2.0]},
    ]

    ranked, meta = rerank_candidates_by_dtw(candidates=candidates, current_curve=[0.0, 1.1, 2.0], top_k=1)

    assert meta["enabled"] is True
    assert ranked[0]["event_id"] == "near"
    assert ranked[0]["dtw_distance"] < 0.2


def test_query_typhoon_events_filters_by_time_and_space(demo_events_json: Path) -> None:
    rows = query_typhoon_events(
        events_json_path=demo_events_json,
        query=QueryBox(
            start_time=datetime(2024, 8, 5),
            end_time=datetime(2024, 8, 6),
            lon_min=118.0,
            lon_max=126.0,
            lat_min=29.0,
            lat_max=37.0,
        ),
        top_k=5,
    )

    assert [row["event_id"] for row in rows] == ["DEMO_202408_WP"]
    assert rows[0]["bbox_overlap_ratio"] > 0
    assert rows[0]["time_overlap_hours"] > 0
    assert rows[0].get("prescreen_score_mode") == "bbox_ratio"
    assert rows[0]["score"] == rows[0]["bbox_overlap_ratio"]


def test_query_typhoon_prescreen_center_distance_mode(demo_events_json: Path) -> None:
    q = QueryBox(
        start_time=datetime(2024, 8, 5),
        end_time=datetime(2024, 8, 6),
        lon_min=118.0,
        lon_max=126.0,
        lat_min=29.0,
        lat_max=37.0,
    )
    rows = query_typhoon_events(
        events_json_path=demo_events_json,
        query=q,
        top_k=5,
        prescreen_score_mode="center_distance",
    )
    assert rows
    assert rows[0]["prescreen_score_mode"] == "center_distance"
    dist = float(rows[0]["center_distance_deg"])
    assert rows[0]["score"] == pytest.approx(1.0 / dist, rel=1e-5)


def test_run_detect_links_demo_typhoon_event(demo_events_json: Path) -> None:
    result = run_detect(
        anomaly_result={
            "start_time": "2024-08-05 00:00:00",
            "end_time": "2024-08-06 00:00:00",
            "lon_min": 118.0,
            "lon_max": 126.0,
            "lat_min": 29.0,
            "lat_max": 37.0,
            "wind_residual": 3.5,
            "wave_residual": 2.8,
            "wind_std": 1.0,
            "wave_std": 1.0,
            "current_curve": [0.5, 1.0, 2.5],
            "wind_dtw_curve": [8.0, 9.0, 10.0],
            "dtw_match_mode": "regional_mean_obs_vs_ibtracs_center",
            "dtw_query_curve": "wind_obs_regional_mean_window",
        },
        events_json_path=str(demo_events_json),
        top_k=3,
    )

    assert result["anomaly_result"]["anomaly_level"] in {"medium", "high"}
    assert result["typhoon_link"]["linked"] is True
    assert result["typhoon_link"]["candidates"][0]["event_id"] == "DEMO_202408_WP"
    dtw = result["typhoon_link"]["retrieval"]["dtw"]
    assert dtw.get("match_mode") == "regional_mean_obs_vs_ibtracs_center"
    assert dtw.get("enabled") is True


def test_infer_typhoon_link_uses_full_history_not_nc_window() -> None:
    defaults = infer_typhoon_link_defaults_from_eddy_result(
        {
            "generated_at": 1_422_000_000,
            "meta": {"nc_path": "data/processed/demo/offline_test_merged_t300.nc"},
        }
    )
    assert defaults.get("history_search_mode") == "full"
    assert defaults.get("anomaly_start_time", "").startswith("2015-")
    assert int(defaults["start_time"][:4]) <= 1950
    assert int(defaults["end_time"][:4]) >= int(str(defaults["anomaly_end_time"])[:4])


def test_rerank_wind_dtw_prefers_ibtracs_track_over_peak_constant() -> None:
    candidates = [
        {
            "event_id": "flat_peak",
            "peak_wind_kt": 100.0,
        },
        {
            "event_id": "shaped_track",
            "wind_track_mps": [0.0, 1.0, 2.0, 3.0],
        },
    ]
    query = [0.0, 1.1, 2.0, 2.9]
    ranked, meta = rerank_candidates_by_dtw(candidates=candidates, current_curve=query, top_k=2)
    assert meta["enabled"] is True
    assert meta["n_candidates_with_track"] == 1
    assert ranked[0]["event_id"] == "shaped_track"
