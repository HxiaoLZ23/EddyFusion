"""
论文 §6.3 系统功能测试 T1～T8（React+FastAPI 演示链路）。

用例编号与 docs/开发规划/答辩演示脚本.md 一致。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web_api.main import app

pytestmark = pytest.mark.paper_system

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel_repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _eddy_weights_available() -> bool:
    from app.services.eddy_demo_service import default_eddy_weight_path_for_stack
    from src.utils.config import resolve_path

    return resolve_path(default_eddy_weight_path_for_stack("3ch")).is_file()


def _client() -> TestClient:
    return TestClient(app)


def test_T1_preprocess_meta_probe(demo_eddy_nc: Path) -> None:
    """T1：上传 NC + meta 探测 → time_len、变量列表。"""
    rel = _rel_repo_path(demo_eddy_nc)
    with _client() as client:
        r = client.get("/api/preprocess/meta", params={"nc_path": rel})
    assert r.status_code == 200
    meta = r.json()
    assert meta.get("time_len") == 8
    assert "variables" in meta
    assert meta.get("variable_map", {}).get("eddy_ready") is True


def test_T2_preprocess_subset_roi(demo_eddy_nc: Path) -> None:
    """T2：ROI/时间裁剪 → 子集 NC 写入 subsets 目录。"""
    rel = _rel_repo_path(demo_eddy_nc)
    with _client() as client:
        r = client.post(
            "/api/preprocess/subset",
            json={
                "nc_path": rel,
                "time_start": 0,
                "time_stop": 3,
                "lon_min": 118.0,
                "lon_max": 125.0,
                "lat_min": 22.0,
                "lat_max": 32.0,
                "task": "eddy",
            },
        )
    assert r.status_code == 200
    out = r.json()
    assert out.get("status") == "ok"
    subset_rel = out.get("nc_path")
    assert isinstance(subset_rel, str) and "subsets/subset_" in subset_rel
    subset_abs = Path(__file__).resolve().parents[1] / subset_rel
    assert subset_abs.is_file()
    try:
        subset_abs.unlink(missing_ok=True)
    except OSError:
        pass


def test_T3_eddy_preview_frame(demo_eddy_nc: Path) -> None:
    """T3：涡旋 preview-frame → 帧图 data URL + stats_rows。"""
    rel = _rel_repo_path(demo_eddy_nc)
    with _client() as client:
        r = client.post("/api/eddy/preview-frame", json={"nc_path": rel, "time_index": 0})
    assert r.status_code == 200
    out = r.json()
    assert out.get("status") == "ok"
    assert out.get("image_data_url", "").startswith("data:image/png;base64,")
    assert isinstance(out.get("stats_rows"), list)
    assert out.get("source") in {"yolo", "adt_fallback"}


@pytest.mark.skipif(not _eddy_weights_available(), reason="本地无 3ch 涡旋权重，T4 跳过（需 outputs/eddy_v6_b0_fair/best.pt 等）")
def test_T4_eddy_dual_mp4_staged_job(demo_eddy_nc: Path) -> None:
    """T4：双路 MP4 异步分阶段 → 底图就绪后完成标注。"""
    rel = _rel_repo_path(demo_eddy_nc)
    with _client() as client:
        created = client.post(
            "/api/jobs",
            json={"type": "eddy_dual_mp4", "nc_path": rel, "fps": 1, "max_frames": 6},
        )
        assert created.status_code == 200
        job_id = created.json()["job_id"]

        deadline = time.time() + 300
        last = None
        while time.time() < deadline:
            st = client.get(f"/api/jobs/{job_id}")
            assert st.status_code == 200
            last = st.json()["job"]
            if last["status"] in ("done", "failed"):
                break
            time.sleep(1.5)

    assert last is not None
    assert last["status"] == "done", last.get("error") or last.get("message")
    result = last.get("result") or {}
    assert result.get("preview_base") or result.get("base_mp4")
    assert result.get("preview_annotated") or result.get("annotated_mp4")


def test_T5_windwave_forecast(demo_windwave_nc: Path) -> None:
    """T5：风浪 forecast → 曲线 + anomaly_segments + Top-K 字段。"""
    rel = _rel_repo_path(demo_windwave_nc)
    with _client() as client:
        r = client.post("/api/windwave/forecast", json={"nc_path": rel, "top_k": 3})
    assert r.status_code == 200
    out = r.json()
    assert out.get("status") == "success"
    assert len(out.get("series") or []) >= 2
    assert isinstance(out.get("anomaly_segments"), list)
    assert out.get("anomaly_level") in {"low", "medium", "high", "unknown"}
    assert "typhoon_candidates" in out
    retrieval = out.get("typhoon_retrieval") or {}
    dtw = (retrieval.get("dtw") if isinstance(retrieval, dict) else None) or {}
    assert dtw.get("match_mode") == "regional_mean_obs_vs_ibtracs_center"


def test_T6_report_save_and_history(demo_windwave_nc: Path) -> None:
    """T6：结构化报告 save/history → 可列表、可读单条。"""
    rel = _rel_repo_path(demo_windwave_nc)
    with _client() as client:
        structured = client.post("/api/report/structured", json={"nc_path": rel, "top_k": 3})
        assert structured.status_code == 200
        md = structured.json().get("markdown")
        assert isinstance(md, str) and md

        saved = client.post(
            "/api/report/save",
            json={
                "nc_path": rel,
                "markdown": md,
                "fields": structured.json().get("fields"),
                "source": "windwave",
                "mode": "offline",
                "title": "论文系统测试报告",
            },
        )
        assert saved.status_code == 200
        rid = saved.json()["id"]

        hist = client.get("/api/report/history", params={"limit": 20})
        assert hist.status_code == 200
        ids = [x["id"] for x in hist.json().get("reports") or []]
        assert rid in ids

        one = client.get(f"/api/report/{rid}")
        assert one.status_code == 200
        assert one.json()["report"]["markdown"] == md


def test_T7_async_job_windwave_forecast(demo_windwave_nc: Path) -> None:
    """T7：异步 job → 提交后进度至 done。"""
    rel = _rel_repo_path(demo_windwave_nc)
    with _client() as client:
        created = client.post(
            "/api/jobs",
            json={"type": "windwave_forecast", "nc_path": rel, "top_k": 3},
        )
        assert created.status_code == 200
        job_id = created.json()["job_id"]

        deadline = time.time() + 120
        last = None
        while time.time() < deadline:
            st = client.get(f"/api/jobs/{job_id}")
            assert st.status_code == 200
            last = st.json()["job"]
            if last["status"] in ("done", "failed"):
                break
            assert last.get("progress", 0) >= 0
            time.sleep(0.8)

    assert last is not None
    assert last["status"] == "done", last.get("error")
    result = last.get("result") or {}
    assert result.get("series")
    retrieval = result.get("typhoon_retrieval") or {}
    dtw = (retrieval.get("dtw") if isinstance(retrieval, dict) else None) or {}
    assert dtw.get("match_mode") == "regional_mean_obs_vs_ibtracs_center"


def test_T8_realtime_connector_status() -> None:
    """T8：准实时 status → 连接器字段完整。"""
    with _client() as client:
        r = client.get("/api/realtime/status")
    assert r.status_code == 200
    out = r.json()
    assert "connected" in out
    assert "poll_dir" in out
    assert "source" in out
    if out.get("connected"):
        assert "nc_count" in out
