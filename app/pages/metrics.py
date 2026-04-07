from __future__ import annotations

from typing import Any

import streamlit as st

from services.metrics_service import MetricsService


def _flatten_numeric(raw: dict[str, Any], prefix: str = "") -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for k, v in raw.items():
        nk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            items.append((nk, float(v)))
        elif isinstance(v, dict):
            items.extend(_flatten_numeric(v, prefix=nk))
    return items


def render(*, metrics_service: MetricsService) -> None:
    st.title("指标看板")
    st.caption("读取 outputs 下现有指标 JSON；缺失时自动降级展示。")

    data = metrics_service.load_all()
    c1, c2, c3 = st.columns(3)
    col_map = [c1, c2, c3]
    for idx, (name, item) in enumerate(data.items()):
        col = col_map[idx % 3]
        with col:
            st.markdown(f"### {name}")
            if item.exists:
                st.success("读取成功")
                pairs = _flatten_numeric(item.raw)
                if pairs:
                    table_data = [{"指标": k, "数值": round(v, 6)} for k, v in pairs]
                    st.dataframe(table_data, use_container_width=True, hide_index=True)
                else:
                    st.caption("该文件无可解析的数值指标")
                with st.expander("原始 JSON", expanded=False):
                    st.json(item.raw)
            else:
                st.warning("文件不存在或不可读")
                st.caption(item.message)

