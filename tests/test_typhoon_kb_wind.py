from __future__ import annotations

from src.anomaly.typhoon_kb import _wind_kt_from_row


def test_wind_kt_falls_back_when_wmo_is_blank_space() -> None:
    row = {"WMO_WIND": " ", "USA_WIND": "45"}
    assert _wind_kt_from_row(row) == 45.0


def test_wind_kt_prefers_wmo_when_valid() -> None:
    row = {"WMO_WIND": "50", "USA_WIND": "45"}
    assert _wind_kt_from_row(row) == 50.0


def test_wind_kt_uses_neumann_for_historical_rows() -> None:
    row = {"WMO_WIND": " ", "USA_WIND": " ", "NEUMANN_WIND": "7"}
    assert _wind_kt_from_row(row) == 7.0
