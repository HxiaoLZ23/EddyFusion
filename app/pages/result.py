from __future__ import annotations

import streamlit as st

from pages.windwave_panel import render_typhoon_linked_outputs
from services.nc_ingest_service import (
    MAX_NC_UPLOAD_MB,
    allowed_nc_suffixes_text,
    cleanup_old_nc_uploads,
    save_uploaded_nc,
)
from src.anomaly.detect import run_detect
from src.anomaly.eddy_typhoon_bridge import build_anomaly_result_for_detect, infer_typhoon_link_defaults_from_eddy_result
from src.anomaly.windwave_nc_bridge import build_eddy_result_from_windwave_netcdf


def _typhoon_link_context_fingerprint(result: dict) -> str:
    import hashlib
    import json

    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    blob = {
        "nc_path": str(meta.get("nc_path") or ""),
        "generated_at": result.get("generated_at"),
        "source_type": result.get("source_type"),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:32]


def _render_typhoon_linkage(result: dict) -> None:
    st.subheader("台风检索")
    fp = _typhoon_link_context_fingerprint(result)
    if st.session_state.get("_ty_link_result_fp") != fp:
        auto_query = infer_typhoon_link_defaults_from_eddy_result(result)
        st.session_state["_ty_link_result_fp"] = fp
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
            st.session_state[key] = value

    with st.expander("参数", expanded=False):
        if st.button("恢复自动推断", key="ty_link_reset_auto"):
            refreshed = infer_typhoon_link_defaults_from_eddy_result(result)
            st.session_state["ty_link_auto_defaults"] = refreshed
            st.session_state["ty_link_start_time"] = refreshed["start_time"]
            st.session_state["ty_link_end_time"] = refreshed["end_time"]
            st.session_state["ty_link_top_k"] = int(refreshed["top_k"])
            st.session_state["ty_link_lon_min"] = float(refreshed["lon_min"])
            st.session_state["ty_link_lon_max"] = float(refreshed["lon_max"])
            st.session_state["ty_link_lat_min"] = float(refreshed["lat_min"])
            st.session_state["ty_link_lat_max"] = float(refreshed["lat_max"])
            st.session_state["ty_link_events_json"] = str(refreshed["events_json_path"])
            st.rerun()
        defaults = st.session_state["ty_link_auto_defaults"]
        c1, c2 = st.columns(2)
        with c1:
            start_time = st.text_input("开始时间", value=defaults["start_time"], key="ty_link_start_time")
            end_time = st.text_input("结束时间", value=defaults["end_time"], key="ty_link_end_time")
            top_k = st.slider("Top-K", min_value=1, max_value=25, value=int(defaults["top_k"]), key="ty_link_top_k")
        with c2:
            lon_min = st.number_input("lon_min", value=float(defaults["lon_min"]), step=0.1, key="ty_link_lon_min")
            lon_max = st.number_input("lon_max", value=float(defaults["lon_max"]), step=0.1, key="ty_link_lon_max")
            lat_min = st.number_input("lat_min", value=float(defaults["lat_min"]), step=0.1, key="ty_link_lat_min")
            lat_max = st.number_input("lat_max", value=float(defaults["lat_max"]), step=0.1, key="ty_link_lat_max")
        events_json_path = st.text_input(
            "事件索引 JSON",
            value=str(defaults["events_json_path"]),
            key="ty_link_events_json",
        )

    anomaly_result = build_anomaly_result_for_detect(
        result,
        link_defaults={
            "start_time": start_time,
            "end_time": end_time,
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "top_k": int(top_k),
            "events_json_path": str(events_json_path),
        },
    )

    try:
        linked = run_detect(
            anomaly_result=anomaly_result,
            auto_link_typhoon=True,
            events_json_path=events_json_path,
            top_k=int(top_k),
        )
    except Exception as e:
        st.error(f"检索失败：{e}")
        return

    render_typhoon_linked_outputs(
        linked,
        key_prefix="ty",
        kb_jump_params={
            "start_time": start_time,
            "end_time": end_time,
            "top_k": int(top_k),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "events_json_path": str(events_json_path),
        },
    )


def render() -> None:
    st.title("风浪预警")
    st.caption("与「涡旋识别」共用会话键 `eddy_last_result`（后写覆盖）。")

    st.subheader("上传 NC")
    st.caption(f"{allowed_nc_suffixes_text()}，≤{MAX_NC_UPLOAD_MB}MB。需 u10/v10 或有效波高等。")
    ww_nc = st.file_uploader("风浪 NC", type=["nc", "nc4", "cdf"], key="windwave_page_nc_uploader")
    if st.button("构建会话", type="primary", key="windwave_nc_build"):
        if ww_nc is None:
            st.warning("请选择文件。")
        else:
            try:
                saved, _tid = save_uploaded_nc(ww_nc)
                cleanup_old_nc_uploads(max_files=30)
                st.session_state["eddy_last_result"] = build_eddy_result_from_windwave_netcdf(saved)
                st.session_state.pop("last_result", None)
                st.success("已加载。")
            except Exception as e:
                st.error(f"解析失败：{e}")

    result = st.session_state.get("eddy_last_result")
    if not result:
        result = st.session_state.get("last_result")
    if not result:
        st.info("请上传 NC，或先在「涡旋识别」完成检测。")
        return

    st.subheader("概览")
    r1, r2, r3 = st.columns(3)
    r1.metric("状态", str(result.get("status", "unknown")))
    r2.metric("来源", str(result.get("source_type", "upload")))
    peak = result.get("peak_score")
    if isinstance(peak, (int, float)):
        r3.metric("峰值分数", f"{float(peak):.3f}")
    else:
        r3.metric("峰值分数", "N/A")

    st.subheader("时间轴")
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
        st.caption("无")

    previews = result.get("preview_images", [])
    if isinstance(previews, list) and previews:
        st.subheader("关键帧")
        cols = st.columns(3)
        for i, p in enumerate(previews[:9]):
            cols[i % 3].image(str(p), caption=f"帧 {i}", use_container_width=True)

    if str(result.get("status", "")).lower() == "success":
        _render_typhoon_linkage(result)

    with st.expander("JSON", expanded=False):
        st.json(result)
