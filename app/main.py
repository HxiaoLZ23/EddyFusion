from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 保证从 app 目录或仓库根目录启动时都能导入 src/*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pages import eddy, hydro, metrics, overview, realtime, result, typhoon_kb, upload
from services.hydro_inference_service import HydroInferenceService
from services.inference_service import build_inference_service
from services.metrics_service import MetricsService


def _init_state() -> None:
    defaults = {
        "task_id": None,
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "uploaded_video_meta": {},
        "last_result": None,
        "task_status": "idle",
        "realtime_running": False,
        "realtime_task_id": None,
        "realtime_last_result": None,
        "realtime_history": [],
        "hydro_last_result": None,
        "hydro_last_preset": "l2",
        "eddy_last_result": None,
        "nav_page": "总览",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    st.set_page_config(page_title="EddyFusion 演示系统", page_icon="🌊", layout="wide")
    _init_state()

    st.sidebar.title("EddyFusion Demo")
    pages = ("总览", "上传", "涡旋识别", "水文推理", "结果", "台风知识库", "实时输入", "指标看板")
    if st.session_state.get("nav_page") not in pages:
        st.session_state["nav_page"] = "总览"
    page = st.sidebar.radio("页面", pages, key="nav_page")
    st.sidebar.caption(f"项目根目录: {Path(__file__).resolve().parents[1]}")

    metrics_service = MetricsService()
    inference_service = build_inference_service(mode="real")
    hydro_service = HydroInferenceService()

    if page == "总览":
        overview.render(metrics_service=metrics_service)
    elif page == "水文推理":
        hydro.render(service=hydro_service)
    elif page == "上传":
        upload.render()
    elif page == "涡旋识别":
        eddy.render()
    elif page == "结果":
        result.render()
    elif page == "台风知识库":
        typhoon_kb.render()
    elif page == "实时输入":
        realtime.render(inference_service=inference_service)
    else:
        metrics.render(metrics_service=metrics_service)


if __name__ == "__main__":
    main()

