from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from src.anomaly.detect import run_detect
from src.anomaly.report import render_report
from src.utils.config import load_yaml, resolve_path


def _safe_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _infer_auto_typhoon_query(result: dict) -> dict:
    data_cfg = {}
    demo_cfg = {}
    try:
        data_cfg = load_yaml("config/data.yaml")
    except Exception:
        data_cfg = {}
    try:
        demo_cfg = load_yaml("app/config/demo.yaml")
    except Exception:
        demo_cfg = {}

    spatial = data_cfg.get("spatial", {}) if isinstance(data_cfg, dict) else {}
    ty_cfg = demo_cfg.get("typhoon_link", {}) if isinstance(demo_cfg, dict) else {}

    lon_min = _safe_float(spatial.get("lon_min"), 117.0)
    lon_max = _safe_float(spatial.get("lon_max"), 127.0)
    lat_min = _safe_float(spatial.get("lat_min"), 31.0)
    lat_max = _safe_float(spatial.get("lat_max"), 41.0)

    generated_at = result.get("generated_at")
    if isinstance(generated_at, (int, float)):
        end_dt = datetime.fromtimestamp(float(generated_at))
    else:
        end_dt = datetime.now()
    window_hours = int(_safe_float(ty_cfg.get("default_window_hours"), 24 * 10))
    start_dt = end_dt - timedelta(hours=max(1, window_hours))

    default_top_k = int(_safe_float(ty_cfg.get("default_top_k"), 5))
    events_json_path = str(
        ty_cfg.get("events_json_path") or resolve_path("data/processed/anomaly/typhoon_kb/events.json")
    )
    return {
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "top_k": max(1, min(default_top_k, 20)),
        "events_json_path": events_json_path,
    }


def _render_typhoon_linkage(result: dict) -> None:
    st.subheader("台风候选事件联动")
    st.caption("已自动推断时间窗与海域范围，并自动检索台风知识库候选事件。")

    auto_query = _infer_auto_typhoon_query(result)
    if "ty_link_auto_defaults" not in st.session_state:
        st.session_state["ty_link_auto_defaults"] = auto_query
    for key, value in (
        ("ty_link_start_time", auto_query["start_time"]),
        ("ty_link_end_time", auto_query["end_time"]),
        ("ty_link_top_k", int(auto_query["top_k"])),
        ("ty_link_lon_min", float(auto_query["lon_min"])),
        ("ty_link_lon_max", float(auto_query["lon_max"])),
        ("ty_link_lat_min", float(auto_query["lat_min"])),
        ("ty_link_lat_max", float(auto_query["lat_max"])),
        ("ty_link_events_json", str(auto_query["events_json_path"])),
    ):
        if key not in st.session_state:
            st.session_state[key] = value

    with st.expander("联动参数（可调整）", expanded=False):
        if st.button("重置为自动推断参数", key="ty_link_reset_auto"):
            refreshed = _infer_auto_typhoon_query(result)
            st.session_state["ty_link_auto_defaults"] = refreshed
            st.session_state["ty_link_start_time"] = refreshed["start_time"]
            st.session_state["ty_link_end_time"] = refreshed["end_time"]
            st.session_state["ty_link_top_k"] = int(refreshed["top_k"])
            st.session_state["ty_link_lon_min"] = float(refreshed["lon_min"])
            st.session_state["ty_link_lon_max"] = float(refreshed["lon_max"])
            st.session_state["ty_link_lat_min"] = float(refreshed["lat_min"])
            st.session_state["ty_link_lat_max"] = float(refreshed["lat_max"])
            st.session_state["ty_link_events_json"] = str(refreshed["events_json_path"])
        defaults = st.session_state["ty_link_auto_defaults"]
        c1, c2 = st.columns(2)
        with c1:
            start_time = st.text_input("开始时间", value=defaults["start_time"], key="ty_link_start_time")
            end_time = st.text_input("结束时间", value=defaults["end_time"], key="ty_link_end_time")
            top_k = st.slider(
                "候选数量 Top-K",
                min_value=1,
                max_value=20,
                value=int(defaults["top_k"]),
                key="ty_link_top_k",
            )
        with c2:
            lon_min = st.number_input("lon_min", value=float(defaults["lon_min"]), step=0.1, key="ty_link_lon_min")
            lon_max = st.number_input("lon_max", value=float(defaults["lon_max"]), step=0.1, key="ty_link_lon_max")
            lat_min = st.number_input("lat_min", value=float(defaults["lat_min"]), step=0.1, key="ty_link_lat_min")
            lat_max = st.number_input("lat_max", value=float(defaults["lat_max"]), step=0.1, key="ty_link_lat_max")
        events_json_path = st.text_input(
            "台风事件索引路径",
            value=str(defaults["events_json_path"]),
            key="ty_link_events_json",
        )

    anomaly_result = {
        "start_time": start_time,
        "end_time": end_time,
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "peak_score": result.get("peak_score"),
        "current_curve": [
            float(it.get("score", 0.0))
            for it in (result.get("timeline") or [])
            if isinstance(it, dict)
        ],
    }

    try:
        linked = run_detect(
            anomaly_result=anomaly_result,
            auto_link_typhoon=True,
            events_json_path=events_json_path,
            top_k=int(top_k),
        )
    except Exception as e:
        st.error(f"台风联动失败：{e}")
        return

    link = linked.get("typhoon_link", {})
    candidates = link.get("candidates", [])
    if not candidates:
        st.warning("未检索到候选台风事件。")
        with st.expander("联动详情（调试）", expanded=False):
            st.json(link)
        return

    st.success(f"检索到 {len(candidates)} 个候选事件")
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
    jump_col1, jump_col2 = st.columns([2, 3])
    with jump_col1:
        if st.button("跳转到台风知识库页（带入当前参数）", key="jump_to_kb_btn", type="secondary"):
            st.session_state["kb_start_time"] = str(start_time)
            st.session_state["kb_end_time"] = str(end_time)
            st.session_state["kb_top_k"] = int(top_k)
            st.session_state["kb_lon_min"] = float(lon_min)
            st.session_state["kb_lon_max"] = float(lon_max)
            st.session_state["kb_lat_min"] = float(lat_min)
            st.session_state["kb_lat_max"] = float(lat_max)
            st.session_state["kb_events_json_path"] = str(events_json_path)
            st.session_state["kb_events_browser_path"] = str(events_json_path)
            st.session_state["kb_query_autorun"] = True
            st.session_state["nav_page"] = "台风知识库"
            st.rerun()
    with jump_col2:
        st.caption("将当前时间窗、海域范围、Top-K 与索引路径同步到“台风知识库”页面。")
    with st.expander("联动详情（技术）", expanded=False):
        st.json(link)
    with st.expander("结构化预警报告", expanded=True):
        report_text = render_report(detect_output=linked)
        st.text(report_text)


def render() -> None:
    st.title("结果联动")
    st.caption("本页展示“涡旋识别”页已生成的结果，并进行台风知识库联动。")
    result = st.session_state.get("eddy_last_result")
    if not result:
        # 兼容旧会话：若曾在结果页直接跑过，仍可展示
        result = st.session_state.get("last_result")
    if not result:
        st.info("尚无可展示结果，请先在“涡旋识别”页面运行真实推理。")
        return

    st.subheader("结果概览")
    r1, r2, r3 = st.columns(3)
    r1.metric("状态", str(result.get("status", "unknown")))
    r2.metric("来源", str(result.get("source_type", "upload")))
    peak = result.get("peak_score")
    if isinstance(peak, (int, float)):
        r3.metric("峰值分数", f"{float(peak):.3f}")
    else:
        r3.metric("峰值分数", "N/A")
    if result.get("summary"):
        st.info(str(result.get("summary")))

    st.subheader("事件时间轴")
    timeline = result.get("timeline", [])
    if timeline:
        rows = []
        for it in timeline:
            score = it.get("score", 0.0)
            rows.append(
                {
                    "时间": it.get("time", "-"),
                    "事件": it.get("event", "-"),
                    "分数": round(float(score), 4) if isinstance(score, (int, float)) else score,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("暂无时间轴事件")

    previews = result.get("preview_images", [])
    if isinstance(previews, list) and previews:
        st.subheader("涡旋关键帧")
        cols = st.columns(3)
        for i, p in enumerate(previews[:9]):
            cols[i % 3].image(str(p), caption=f"eddy_{i:02d}", use_container_width=True)

    if str(result.get("status", "")).lower() == "success":
        _render_typhoon_linkage(result)

    with st.expander("查看原始结果 JSON", expanded=False):
        st.json(result)

