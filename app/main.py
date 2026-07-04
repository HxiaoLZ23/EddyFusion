from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 保证从 app 目录或仓库根目录启动时都能找到 src/*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from pages import eddy, metrics, overview, result, typhoon_kb
from services.metrics_service import MetricsService

# 单模块精简导航：无实时/离线/水文
PAGES: tuple[str, ...] = (
    "总览",
    "涡旋识别",
    "风浪预警",
    "台风查询",
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
        "实时输入": "总览",
        "实时系统": "总览",
        "离线系统": "总览",
        "水文推理": "总览",
    }
    if legacy in mapping:
        st.session_state["nav_page"] = mapping[legacy]


def main() -> None:
    st.set_page_config(
        page_title="海洋环境演示（涡旋 / 风浪 / 台风）",
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

    st.sidebar.title("导航")
    page = st.sidebar.radio("页面", PAGES, key="nav_page")

    metrics_service = MetricsService()

    if page == "总览":
        overview.render(metrics_service=metrics_service)
    elif page == "涡旋识别":
        eddy.render()
    elif page == "风浪预警":
        result.render()
    elif page == "台风查询":
        typhoon_kb.render()
    else:
        metrics.render(metrics_service=metrics_service)


if __name__ == "__main__":
    main()
