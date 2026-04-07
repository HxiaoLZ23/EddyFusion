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
    st.title("海洋环境智能分析演示")
    st.write("本演示系统用于展示三模块能力、视频输入流程与阶段性指标。")

    c1, c2, c3 = st.columns(3)
    c1.metric("涡旋识别", "准备中", "数据管线对接")
    c2.metric("水文预测", "可用", "已支持 val/test 指标读取")
    c3.metric("风浪异常", "可用(演示)", "支持视频上传与 mock 结果")

    st.subheader("最近指标摘要")
    summary = metrics_service.load_all()
    cols = st.columns(3)
    for i, (name, data) in enumerate(summary.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            if data.exists:
                st.success("已读取")
                nums = _numeric_metrics(data.raw)
                if nums:
                    top_items = list(nums.items())[:3]
                    for mk, mv in top_items:
                        st.metric(mk, f"{mv:.4f}")
                else:
                    st.caption("无可展示的数值指标")
                with st.expander("查看原始明细", expanded=False):
                    st.json(data.raw)
            else:
                st.warning("待生成")
                st.caption(data.message)

    st.info("建议演示路径：上传视频 -> 查看结果 -> 回到指标看板说明当前模型状态。")

