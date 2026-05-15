"""台风联动结果展示：候选表、规则报告、可选大模型解读（风浪预警页与实时系统复用）。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.anomaly.llm_report import build_user_payload_from_detect, payload_fingerprint, try_llm_report
from src.anomaly.report import render_report
from src.utils.config import load_yaml


def render_typhoon_linked_outputs(
    linked: dict[str, Any],
    *,
    key_prefix: str,
    kb_jump_params: dict[str, Any] | None = None,
    llm_caption_extra: str = "",
) -> None:
    link = linked.get("typhoon_link", {})
    candidates = link.get("candidates", [])
    if not candidates:
        st.warning("未检索到候选台风事件。下方仍可使用规则模板与智能解读（基于当前异常评估与空候选）。")
    else:
        st.success(f"检索到 {len(candidates)} 个候选事件")

    if candidates:
        rows = []
        for c in candidates:
            rows.append(
                {
                    "事件ID": c.get("event_id", "-"),
                    "名称": c.get("name", ""),
                    "开始时间": c.get("start_time", ""),
                    "结束时间": c.get("end_time", ""),
                    "强度级别": c.get("intensity_level", ""),
                    "峰值风速(kt)": c.get("peak_wind_kt", ""),
                    "时窗重叠(h)": c.get("time_overlap_hours", ""),
                    "区域重叠": c.get("bbox_overlap_ratio", ""),
                    "分数": c.get("score", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if kb_jump_params:
        jump_col1, jump_col2 = st.columns([2, 3])
        with jump_col1:
            if st.button(
                "跳转到台风知识库页（带入当前参数）",
                key=f"{key_prefix}_jump_to_kb_btn",
                type="secondary",
                disabled=not bool(candidates),
            ):
                st.session_state["kb_start_time"] = str(kb_jump_params["start_time"])
                st.session_state["kb_end_time"] = str(kb_jump_params["end_time"])
                st.session_state["kb_top_k"] = int(kb_jump_params["top_k"])
                st.session_state["kb_lon_min"] = float(kb_jump_params["lon_min"])
                st.session_state["kb_lon_max"] = float(kb_jump_params["lon_max"])
                st.session_state["kb_lat_min"] = float(kb_jump_params["lat_min"])
                st.session_state["kb_lat_max"] = float(kb_jump_params["lat_max"])
                st.session_state["kb_events_json_path"] = str(kb_jump_params["events_json_path"])
                st.session_state["kb_events_browser_path"] = str(kb_jump_params["events_json_path"])
                st.session_state["kb_query_autorun"] = True
                st.session_state["_nav_page_pending"] = "台风知识库"
                st.rerun()
        with jump_col2:
            st.caption("将当前时间窗、海域范围、Top-K 与索引路径同步到“台风知识库”页面。")

    with st.expander("联动详情（技术）", expanded=False):
        st.json(link)
    with st.expander("结构化预警报告（规则模板）", expanded=True):
        st.text(render_report(detect_output=linked))

    llm_cfg: dict[str, Any] = {}
    try:
        demo_full = load_yaml("app/config/demo.yaml")
        if isinstance(demo_full, dict):
            llm_cfg = demo_full.get("llm_report") or {}
    except Exception:
        pass
    if not bool(llm_cfg.get("show_panel", True)):
        return

    fp_link = payload_fingerprint(build_user_payload_from_detect(linked))
    cache_key = f"llm_report_cache_{key_prefix}_{fp_link}"
    max_tok = int(float(llm_cfg.get("max_tokens", 2048) or 2048))
    expand_key = f"_llm_report_expand_next_{key_prefix}"
    _llm_expand = bool(st.session_state.pop(expand_key, False))

    cap = (
        "基于当前联动结果 JSON 调用百炼部署模型；密钥与模型名从环境变量读取，勿写入代码。"
        + (" " + llm_caption_extra if llm_caption_extra else "")
    )
    with st.expander("智能解读报告（大模型 · 可选）", expanded=_llm_expand):
        st.caption(cap)
        model_override = st.text_input(
            "覆盖模型部署代号（可选，默认 DASHSCOPE_MODEL）",
            value="",
            key=f"{key_prefix}_llm_model_override",
            placeholder="留空则使用环境变量",
        )
        if st.button("生成智能解读", type="primary", key=f"{key_prefix}_llm_gen_btn"):
            with st.spinner("调用大模型中…"):
                parsed, err, _fp = try_llm_report(
                    linked,
                    model=(model_override or "").strip() or None,
                    max_tokens=max_tok,
                )
            if parsed:
                st.session_state[cache_key] = {"parsed": parsed, "error": ""}
                st.session_state[expand_key] = True
                st.rerun()
            else:
                st.session_state[cache_key] = {"parsed": None, "error": err}
                st.error((err or "未知错误")[:4000])

        cached = st.session_state.get(cache_key)
        if isinstance(cached, dict) and cached.get("parsed"):
            p = cached["parsed"]
            st.markdown("##### 综述（是否异常）")
            st.markdown(str(p.get("summary_anomaly", "")))
            st.markdown("##### 影响判断")
            st.markdown(str(p.get("impact", "")))
            st.markdown("##### 历史类比")
            st.markdown(str(p.get("historical_analogy", "")))
            st.markdown("##### 建议动作")
            for i, a in enumerate(p.get("actions") or [], start=1):
                st.markdown(f"{i}. {a}")
        elif isinstance(cached, dict) and cached.get("error"):
            st.warning(str(cached["error"])[:4000])
