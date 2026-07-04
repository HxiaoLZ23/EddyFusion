from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from web_api.main import app


def test_health_and_eddy_ping_endpoints() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        eddy = client.get("/api/eddy/ping")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert eddy.status_code == 200
    assert eddy.json()["ok"] is True


def test_typhoon_kb_status_and_query(demo_events_json: Path) -> None:
    with TestClient(app) as client:
        status = client.get("/api/typhoon-kb/status")
        query = client.post(
            "/api/typhoon-kb/query",
            json={
                "start_time": "2024-08-05 00:00:00",
                "end_time": "2024-08-06 00:00:00",
                "lon_min": 118.0,
                "lon_max": 126.0,
                "lat_min": 29.0,
                "lat_max": 37.0,
                "top_k": 5,
                "events_json_path": str(demo_events_json),
            },
        )

    assert status.status_code == 200
    assert "ready" in status.json()
    assert query.status_code == 200
    payload = query.json()
    assert payload["status"] == "success"
    assert payload["count"] == 1
    assert payload["candidates"][0]["event_id"] == "DEMO_202408_WP"


def test_windwave_offline_report_endpoint_uses_demo_netcdf(demo_windwave_nc: Path) -> None:
    rel = demo_windwave_nc.relative_to(Path(__file__).resolve().parents[1]).as_posix()

    with TestClient(app) as client:
        response = client.post("/api/windwave/offline-report", json={"nc_path": rel})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["report_text"], str) and payload["report_text"]
    assert payload["anomaly_level"] in {"low", "medium", "high", "unknown"}
    assert "typhoon_kb_ready" in payload
