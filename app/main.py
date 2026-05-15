from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 保证从 app 目录或仓库根目录启动时都能找到 src/*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pages import eddy, hydro, metrics, offline_system, overview, realtime, result, typhoon_kb
from services.hydro_inference_service import HydroInferenceService
from services.inference_service import build_inference_service
from services.metrics_service import MetricsService

# 侧边栏展示顺序：总览 → 实时/离线系统 → 三模块工作台 → 台风知识库 → 指标看板
PAGES: tuple[str, ...] = (
    "总览",
    "实时系统",
    "离线系统",
    "涡旋识别",
    "水文推理",
    "风浪预警",
    "台风知识库",
    "指标看板",
)


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


def _migrate_legacy_nav() -> None:
    legacy = st.session_state.get("nav_page")
    mapping = {
        "上传": "涡旋识别",
        "结果": "风浪预警",
        "实时输入": "实时系统",
    }
    if legacy in mapping:
        st.session_state["nav_page"] = mapping[legacy]


def main() -> None:
    st.set_page_config(
        page_title="EddyFusion：面向涡旋—水文—风浪的海洋环境智能分析与预警平台",
        page_icon="🌊",
        layout="wide",
    )
    _init_state()
    _migrate_legacy_nav()

    pending = st.session_state.pop("_nav_page_pending", None)
    if pending in PAGES:
        st.session_state["nav_page"] = pending
    elif st.session_state.get("nav_page") not in PAGES:
        st.session_state["nav_page"] = "总览"

    st.sidebar.title("EddyFusion")
    st.sidebar.caption("面向涡旋—水文—风浪的海洋环境智能分析与预警平台")
    st.sidebar.caption("导航：总览 / 实时·离线系统 / 三模块 / 台风知识库定调")
    page = st.sidebar.radio("页面", PAGES, key="nav_page")
    st.sidebar.caption(f"项目根目录: {Path(__file__).resolve().parents[1]}")

    metrics_service = MetricsService()
    inference_service = build_inference_service(mode="real")
    hydro_service = HydroInferenceService()

    if page == "总览":
        overview.render(metrics_service=metrics_service)
    elif page == "水文推理":
        hydro.render(service=hydro_service)
    elif page == "涡旋识别":
        eddy.render()
    elif page == "风浪预警":
        result.render()
    elif page == "台风知识库":
        typhoon_kb.render()
    elif page == "实时系统":
        realtime.render(inference_service=inference_service)
    elif page == "离线系统":
        offline_system.render()
    else:
        metrics.render(metrics_service=metrics_service)


if __name__ == "__main__":
    main()
