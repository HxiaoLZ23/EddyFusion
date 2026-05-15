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
    """会话结果变化（NC 路径或重新生成时间）时刷新联动表单，避免沿用旧 Streamlit session 参数。"""
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
    st.subheader("台风候选事件联动")
    st.caption("已自动推断时间窗与海域范围，并自动检索台风知识库候选事件。")
    if result.get("wind_wave_from_companion_npz"):
        if result.get("wind_wave_from_netcdf"):
            st.info(
                "已使用**本页上传的 NetCDF** 提取风浪时序（obs 与平滑基线 pred），参与 `run_detect` 的 3σ 分级与台风曲线 DTW。"
            )
        else:
            st.info(
                "已启用涡旋页上传的**配套 NPZ** 中 `demo_wind_*` / `demo_wave_*`，参与本页 `run_detect` 的 3σ 分级与台风曲线 DTW。"
            )

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

    with st.expander("联动参数（可调整）", expanded=False):
        if st.button("重置为自动推断参数", key="ty_link_reset_auto"):
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
        defaults = st.session_state["ty_link_auto_defaults"]
        c1, c2 = st.columns(2)
        with c1:
            start_time = st.text_input("开始时间", value=defaults["start_time"], key="ty_link_start_time")
            end_time = st.text_input("结束时间", value=defaults["end_time"], key="ty_link_end_time")
            top_k = st.slider(
                "候选数量 Top-K",
                min_value=1,
                max_value=25,
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
        st.error(f"台风联动失败：{e}")
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
        llm_caption_extra="风浪演示：可在本页直接上传含 u10/v10/浪高的 NC，或使用「涡旋识别」页 NetCDF 路径① 写入的会话；实时系统见峰值阈值+手动生成。",
    )


def render() -> None:
    st.title("风浪预警")
    st.caption(
        "本页可 **直接上传含风浪要素的 NetCDF** 运行 `run_detect` 与台风联动；也可沿用「涡旋识别」页的 NetCDF 会话。"
        " **同一会话键 `eddy_last_result`：以最后一次成功操作为准**——涡旋页「运行检测」、或本页「从 NC 构建预警输入」后写覆盖先写，并非并行两套上下文。"
        " **实时系统**在 `demo.yaml` 的 `realtime_windwave.peak_score_threshold` 达标后，由用户手动点击生成联动与解读。"
    )

    st.subheader("风浪 NetCDF（独立入口）")
    st.caption(
        f"上传 {allowed_nc_suffixes_text()}（单文件 ≤{MAX_NC_UPLOAD_MB}MB），需含 **u10/v10** 或 **有效波高** 等变量，"
        "规则与 `src/preprocess/anomaly_dataset.py` 中 `extract_wind_wave_series_from_netcdf` 一致。"
        " 构建成功后写入本会话的 `eddy_last_result`，与「涡旋识别」页带风浪的 NC 路径等价接入下游。"
    )
    ww_nc = st.file_uploader("选择风浪要素 NC", type=["nc", "nc4", "cdf"], key="windwave_page_nc_uploader")
    if st.button("从 NC 构建预警输入", type="primary", key="windwave_nc_build"):
        if ww_nc is None:
            st.warning("请先选择 NC 文件。")
        else:
            try:
                saved, _tid = save_uploaded_nc(ww_nc)
                cleanup_old_nc_uploads(max_files=30)
                st.session_state["eddy_last_result"] = build_eddy_result_from_windwave_netcdf(saved)
                st.session_state.pop("last_result", None)
                st.success("已从 NC 加载风浪时序，下方展示预警概览与台风联动。")
            except Exception as e:
                st.error(f"NC 解析失败：{e}")

    result = st.session_state.get("eddy_last_result")
    if not result:
        # 兼容旧会话：若曾在结果页直接跑过，仍可展示
        result = st.session_state.get("last_result")
    if not result:
        st.info("尚无可展示结果：请在本页上传风浪 NC，或到「涡旋识别」页面运行推理后再回到本页。")
        return

    st.subheader("预警概览")
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
    stt = str(result.get("source_type", ""))
    if stt == "netcdf_windwave":
        st.caption("当前上下文来自**本页**上传的风浪 NC，无涡旋关键帧属正常。")
    elif stt == "netcdf_eddy_windwave":
        st.caption("当前由**涡旋页** NC（流场+风浪）写入；无视频关键帧属正常。")
    elif stt == "netcdf_eddy_only":
        st.caption("当前为**涡旋页**仅流场 NC：风浪侧可能走演示代理，完整风浪分级建议用路径① 或本页上传风浪 NC。")

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

