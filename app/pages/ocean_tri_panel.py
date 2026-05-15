"""实时/离线「全链路」页共用的三模块占位布局（后续接入真实 NC 驱动推理结果）。"""

from __future__ import annotations

import streamlit as st


def render_tri_module_strip(*, mode_label: str) -> None:
    st.divider()
    st.subheader("三模块同屏预览（建设中）")
    st.caption(
        f"模式：{mode_label}。后续在此嵌入涡旋叠加图、水文热力/曲线、风浪告警摘要；"
        "详见 `相关文件/下一步开发方向_数据入口实时离线与大屏.md` §3。"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 涡旋识别")
        st.info("底图 + 分割/检测框（NC→融合 BGR→YOLO）")
    with c2:
        st.markdown("##### 水文预测")
        st.info("72h+ 要素曲线与栅格热力图")
    with c3:
        st.markdown("##### 风浪预警")
        st.info("异常等级与台风联动摘要")
