from __future__ import annotations

from pathlib import Path

import streamlit as st

from pages import metrics, overview, realtime, result, upload
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    st.set_page_config(page_title="EddyFusion 演示系统", page_icon="🌊", layout="wide")
    _init_state()

    st.sidebar.title("EddyFusion Demo")
    page = st.sidebar.radio("页面", ("总览", "上传", "结果", "实时输入", "指标看板"))
    st.sidebar.caption(f"项目根目录: {Path(__file__).resolve().parents[1]}")

    metrics_service = MetricsService()
    inference_service = build_inference_service(mode="mock")

    if page == "总览":
        overview.render(metrics_service=metrics_service)
    elif page == "上传":
        upload.render()
    elif page == "结果":
        result.render(inference_service=inference_service)
    elif page == "实时输入":
        realtime.render(inference_service=inference_service)
    else:
        metrics.render(metrics_service=metrics_service)


if __name__ == "__main__":
    main()

