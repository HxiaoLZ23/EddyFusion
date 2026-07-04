from __future__ import annotations

from typing import Any

import streamlit as st

from services.metrics_service import MetricsService


def _numeric_metrics(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)):
                    out[f"{k}.{sk}"] = float(sv)
    return out


def render(*, metrics_service: MetricsService) -> None:
    st.title("总览")
    c1, c2, c3 = st.columns(3)
    c1.metric("涡旋识别", "NetCDF → YOLO")
    c2.metric("风浪预警", "序列 + 台风检索")
    c3.metric("台风查询", "IBTrACS 索引")

    st.subheader("指标摘要")
    summary = metrics_service.load_all()
    if not summary:
        st.caption("暂无配置的指标 JSON。")
        return
    cols = st.columns(max(1, len(summary)))
    for i, (name, data) in enumerate(summary.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            if data.exists:
                nums = _numeric_metrics(data.raw)
                if nums:
                    for mk, mv in list(nums.items())[:4]:
                        st.metric(mk, f"{mv:.4f}")
                with st.expander("原始 JSON", expanded=False):
                    st.json(data.raw)
            else:
                st.caption(data.message)

