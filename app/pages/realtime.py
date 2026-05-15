from __future__ import annotations

import time
import uuid
from typing import Any

import cv2
import numpy as np
import streamlit as st

from pages.ocean_tri_panel import render_tri_module_strip
from pages.windwave_panel import render_typhoon_linked_outputs
from services.inference_service import InferenceService
from services.realtime_pipeline import FramePacket, RealtimePipeline
from src.anomaly.detect import run_detect
from src.anomaly.eddy_typhoon_bridge import build_anomaly_result_for_detect, infer_typhoon_link_defaults_from_eddy_result
from src.utils.config import load_yaml


def _get_pipeline(infer_fps: float) -> RealtimePipeline:
    if "realtime_pipeline" not in st.session_state:
        st.session_state["realtime_pipeline"] = RealtimePipeline(queue_maxlen=8, infer_fps=infer_fps)
    pipeline: RealtimePipeline = st.session_state["realtime_pipeline"]
    pipeline.update_infer_fps(infer_fps)
    return pipeline


def _source_key(source_type: str, source_uri: str) -> str:
    return f"{source_type}:{source_uri if source_type == 'rtsp' else 'camera:0'}"


def _release_capture() -> None:
    cap = st.session_state.get("realtime_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state["realtime_cap"] = None
    st.session_state["realtime_source_key"] = None


def _ensure_capture(source_type: str, source_uri: str) -> tuple[bool, str]:
    key = _source_key(source_type, source_uri)
    cap = st.session_state.get("realtime_cap")
    old_key = st.session_state.get("realtime_source_key")
    if cap is not None and old_key == key and cap.isOpened():
        return True, "capture_ready"

    _release_capture()
    src: Any = 0 if source_type == "camera" else source_uri
    new_cap = cv2.VideoCapture(src)
    if not new_cap.isOpened():
        return False, "无法打开视频源"
    st.session_state["realtime_cap"] = new_cap
    st.session_state["realtime_source_key"] = key
    return True, "capture_opened"


def _read_one_frame(source_type: str, source_uri: str) -> tuple[bool, Any]:
    ok_open, msg = _ensure_capture(source_type, source_uri)
    if not ok_open:
        return False, msg
    cap = st.session_state.get("realtime_cap")
    if cap is None:
        return False, "视频源不可用"
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, "读取帧失败"
    return True, frame


def _maybe_reconnect_rtsp(source_uri: str) -> tuple[bool, str]:
    # 指数退避重连：1s -> 2s -> 4s -> 8s（上限 8s）
    now = time.time()
    next_retry = float(st.session_state.get("realtime_next_retry_ts", 0.0))
    if now < next_retry:
        return False, f"等待重连窗口：{next_retry - now:.1f}s"
    attempts = int(st.session_state.get("realtime_reconnect_attempts", 0))
    backoff = min(8.0, 2.0 ** max(0, attempts))
    st.session_state["realtime_reconnect_attempts"] = attempts + 1
    st.session_state["realtime_next_retry_ts"] = now + backoff
    _release_capture()
    ok, msg = _ensure_capture("rtsp", source_uri)
    if ok:
        st.session_state["realtime_reconnect_attempts"] = 0
        st.session_state["realtime_next_retry_ts"] = 0.0
        return True, "RTSP 重连成功"
    return False, f"RTSP 重连失败：{msg}，下次重试约 {backoff:.0f}s 后"


def _render_fixed_frame(frame_slot: Any, frame_bgr: np.ndarray | None, caption: str) -> None:
    """固定画面区域，避免画面出现/消失导致下方布局抖动。"""
    if frame_bgr is None:
        frame_slot.markdown(
            (
                "<div style='height:540px;border:1px dashed #888;border-radius:8px;"
                "display:flex;align-items:center;justify-content:center;color:#999;'>"
                "等待视频帧..."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return
    # 固定输出分辨率，避免每帧尺寸变化造成页面重排
    frame_show = cv2.resize(frame_bgr[:, :, ::-1], (960, 540), interpolation=cv2.INTER_AREA)
    frame_slot.image(frame_show, caption=caption, width=960)


def _realtime_peak_threshold() -> float:
    try:
        demo = load_yaml("app/config/demo.yaml")
        if isinstance(demo, dict):
            rw = demo.get("realtime_windwave") or {}
            if isinstance(rw, dict) and rw.get("peak_score_threshold") is not None:
                return float(rw["peak_score_threshold"])
            mock = demo.get("mock") or {}
            if isinstance(mock, dict) and mock.get("anomaly_threshold") is not None:
                return float(mock["anomaly_threshold"])
    except Exception:
        pass
    return 0.5


def _realtime_result_fingerprint(r: dict[str, Any] | None) -> str:
    if not r:
        return ""
    return f"{r.get('generated_at')}|{r.get('peak_score')}|{len(r.get('timeline') or [])}"


def _render_history_table_fixed(rows: list[dict[str, Any]], n_show: int = 10) -> list[dict[str, str]]:
    out_raw = list(rows[-n_show:])
    out: list[dict[str, str]] = []
    for r in out_raw:
        out.append(
            {
                "time": str(r.get("time", "-")),
                "peak_score": str(r.get("peak_score", "-")),
                "status": str(r.get("status", "-")),
            }
        )
    while len(out) < n_show:
        out.append({"time": "-", "peak_score": "-", "status": "-"})
    return out


def render(*, inference_service: InferenceService) -> None:
    st.title("实时系统")
    st.caption("视频/摄像头或 RTSP + 队列 + 限频涡旋推理；准实时 NC 接入见下方展开区。")

    with st.expander("准实时 NetCDF / 最新格点产品（接入占位）", expanded=False):
        st.caption(
            "落地形态可为：目录轮询最新 `.nc`、对象存储通知、THREDDS/OpenDAP `latest` 等。"
            " 当前未接后端；请先用「离线系统」上传 NC 做元数据摘要，或继续使用下方视频流。"
        )
        poll_dir = st.text_input("轮询目录占位（未启用）", value="", key="rt_nc_poll_dir_placeholder")
        if poll_dir.strip():
            st.info("连接器尚未实现；见 `services/nc_preprocess_facade.py` 与开发方向文档 §2.1。")

    c1, c2 = st.columns(2)
    with c1:
        source_mode = st.selectbox("输入源", options=["camera", "rtsp"], index=0)
    with c2:
        infer_fps = st.slider("推理频率 (fps)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    display_fps = 10

    source_uri = ""
    if source_mode == "rtsp":
        source_uri = st.text_input("RTSP 地址", value="rtsp://127.0.0.1:8554/live")

    if "realtime_running" not in st.session_state:
        st.session_state["realtime_running"] = False
    if "realtime_task_id" not in st.session_state:
        st.session_state["realtime_task_id"] = f"rt-{uuid.uuid4().hex[:8]}"
    if "realtime_last_result" not in st.session_state:
        st.session_state["realtime_last_result"] = None
    if "realtime_history" not in st.session_state:
        st.session_state["realtime_history"] = []
    if "realtime_cap" not in st.session_state:
        st.session_state["realtime_cap"] = None
    if "realtime_source_key" not in st.session_state:
        st.session_state["realtime_source_key"] = None
    if "realtime_reconnect_attempts" not in st.session_state:
        st.session_state["realtime_reconnect_attempts"] = 0
    if "realtime_next_retry_ts" not in st.session_state:
        st.session_state["realtime_next_retry_ts"] = 0.0
    if "realtime_last_frame" not in st.session_state:
        st.session_state["realtime_last_frame"] = None
    if "realtime_windwave_linked" not in st.session_state:
        st.session_state["realtime_windwave_linked"] = None
    if "realtime_windwave_source_fp" not in st.session_state:
        st.session_state["realtime_windwave_source_fp"] = ""

    if "realtime_running_toggle" not in st.session_state:
        st.session_state["realtime_running_toggle"] = bool(st.session_state["realtime_running"])
    op1, op2 = st.columns([2, 1])
    with op1:
        running_toggle = st.toggle("实时运行", key="realtime_running_toggle")
    with op2:
        if st.button("重置会话", help="停止并清理队列与最近结果"):
            st.session_state["realtime_running"] = False
            st.session_state["realtime_running_toggle"] = False
            st.session_state["realtime_last_result"] = None
            st.session_state["realtime_history"] = []
            st.session_state["realtime_last_frame"] = None
            st.session_state["realtime_windwave_linked"] = None
            st.session_state["realtime_windwave_source_fp"] = ""
            _release_capture()

    if running_toggle and not st.session_state["realtime_running"]:
        st.session_state["realtime_running"] = True
        st.session_state["realtime_reconnect_attempts"] = 0
        st.session_state["realtime_next_retry_ts"] = 0.0
    elif (not running_toggle) and st.session_state["realtime_running"]:
        st.session_state["realtime_running"] = False
        _release_capture()

    pipeline = _get_pipeline(infer_fps)

    # 关键：布局只创建一次。后续 fragment tick 仅更新这些占位内容，避免反复重建造成抖动。
    status_box = st.empty()
    st.subheader("实时画面")
    frame_slot = st.empty()
    panel_cols = st.columns([2, 1])
    result_box = panel_cols[0].empty()
    info_box = panel_cols[1].empty()
    alert_box = st.empty()
    st.subheader("实时历史（最近 10 条）")
    history_box = st.empty()

    def _render_live_block() -> None:
        _render_fixed_frame(
            frame_slot,
            st.session_state.get("realtime_last_frame"),
            caption="等待视频帧...",
        )

        status_box.info(
            f"状态: {'running' if st.session_state['realtime_running'] else 'stopped'} | "
            f"queue={pipeline.queue_size()} | infer_fps={infer_fps:.1f} | display_fps={display_fps}(fixed)"
        )

        if not st.session_state["realtime_running"]:
            _render_fixed_frame(frame_slot, st.session_state.get("realtime_last_frame"), caption="最近一帧（已停止）")
            last = st.session_state.get("realtime_last_result")
            with result_box.container():
                st.subheader("实时分析结果")
                c1, c2, c3 = st.columns(3)
                c1.metric("状态", str(last.get("status", "idle")) if last else "idle")
                c2.metric("来源", str(last.get("source_type", "-")) if last else "-")
                peak = last.get("peak_score") if last else None
                c3.metric("峰值", f"{float(peak):.3f}" if isinstance(peak, (int, float)) else "N/A")
            with info_box.container():
                st.subheader("运行信息")
                st.metric("队列长度", pipeline.queue_size())
                st.metric("重连次数", int(st.session_state.get("realtime_reconnect_attempts", 0)))
            with alert_box.container():
                st.caption("状态提示：已停止")
            history_box.table(_render_history_table_fixed(st.session_state["realtime_history"], n_show=10))
            return

        ok, data = _read_one_frame(source_mode, source_uri)
        if not ok:
            if source_mode == "rtsp":
                ok_rec, rec_msg = _maybe_reconnect_rtsp(source_uri)
                if ok_rec:
                    ok2, data2 = _read_one_frame(source_mode, source_uri)
                    if not ok2:
                        _render_fixed_frame(
                            frame_slot,
                            st.session_state.get("realtime_last_frame"),
                            caption="等待视频帧（重连中）",
                        )
                        with alert_box.container():
                            st.warning(f"{rec_msg}，但读取帧仍失败：{data2}")
                        history_box.table(_render_history_table_fixed(st.session_state["realtime_history"], n_show=10))
                        return
                    data = data2
                else:
                    _render_fixed_frame(
                        frame_slot,
                        st.session_state.get("realtime_last_frame"),
                        caption="等待视频帧（重连中）",
                    )
                    with alert_box.container():
                        st.warning(rec_msg)
                    history_box.table(_render_history_table_fixed(st.session_state["realtime_history"], n_show=10))
                    return
            else:
                st.session_state["realtime_running"] = False
                _release_capture()
                _render_fixed_frame(
                    frame_slot,
                    st.session_state.get("realtime_last_frame"),
                    caption="摄像头不可用",
                )
                with alert_box.container():
                    st.error(str(data))
                history_box.table(_render_history_table_fixed(st.session_state["realtime_history"], n_show=10))
                return

        frame = data
        st.session_state["realtime_last_frame"] = frame
        ts = time.time()
        packet = FramePacket(
            frame=frame,
            timestamp=ts,
            source_type=source_mode,
            metadata={"source_uri": source_uri if source_mode == "rtsp" else "camera:0"},
        )
        pipeline.enqueue(packet)
        _render_fixed_frame(
            frame_slot,
            frame,
            caption=f"实时帧 @ {time.strftime('%H:%M:%S', time.localtime(ts))}",
        )

        result = pipeline.maybe_infer(inference_service, task_id=st.session_state["realtime_task_id"])
        if result is not None:
            st.session_state["realtime_last_result"] = result
            ann = result.get("annotated_frame_bgr")
            if ann is not None:
                st.session_state["realtime_last_frame"] = ann
                _render_fixed_frame(
                    frame_slot,
                    ann,
                    caption=f"实时识别框 @ {time.strftime('%H:%M:%S', time.localtime(ts))}",
                )
            hist = st.session_state["realtime_history"]
            hist.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "peak_score": f"{float(result.get('peak_score', 0.0)):.4f}",
                    "status": result.get("status", "unknown"),
                }
            )
            st.session_state["realtime_history"] = hist[-10:]

        last = st.session_state.get("realtime_last_result")
        with result_box.container():
            st.subheader("实时分析结果")
            c1, c2, c3 = st.columns(3)
            c1.metric("状态", str(last.get("status", "running")) if last else "running")
            c2.metric("来源", str(last.get("source_type", source_mode)) if last else source_mode)
            peak = last.get("peak_score") if last else None
            c3.metric("峰值", f"{float(peak):.3f}" if isinstance(peak, (int, float)) else "N/A")

        with info_box.container():
            st.subheader("运行信息")
            st.metric("队列长度", pipeline.queue_size())
            st.metric("重连次数", int(st.session_state.get("realtime_reconnect_attempts", 0)))

        with alert_box.container():
            st.caption("状态提示：运行中")

        history_box.table(_render_history_table_fixed(st.session_state["realtime_history"], n_show=10))

    if hasattr(st, "fragment"):
        if st.session_state.get("realtime_running", False):
            @st.fragment(run_every="100ms")
            def _live_fragment() -> None:
                _render_live_block()
        else:
            @st.fragment()
            def _live_fragment() -> None:
                _render_live_block()
        _live_fragment()
    else:
        st.warning("当前 Streamlit 版本不支持 fragment，实时页可能存在整页刷新抖动。")
        _render_live_block()

    peak_thr = _realtime_peak_threshold()
    st.subheader("风浪联动与解读（手动生成）")
    st.caption(
        f"与「上传视频 → 风浪预警」主路径不同：实时流仅在 **peak_score ≥ {peak_thr:g}** 且推理成功时，"
        "才可点击下方按钮生成台风候选、规则报告与可选大模型解读。"
    )
    last_rt = st.session_state.get("realtime_last_result")
    peak_val: float | None = None
    if last_rt and isinstance(last_rt.get("peak_score"), (int, float)):
        peak_val = float(last_rt["peak_score"])
    ok_status = bool(last_rt) and str(last_rt.get("status", "")).lower() == "success"
    can_gen = ok_status and peak_val is not None and peak_val >= peak_thr
    if not last_rt:
        st.info("尚无实时推理结果，请先开启「实时运行」并等待至少一帧推理完成。")
    elif not ok_status:
        st.warning(f"当前状态为「{last_rt.get('status', 'unknown')}」，需成功完成推理后才可生成联动。")
    elif peak_val is None:
        st.warning("当前结果缺少 peak_score，无法判断阈值。")
    elif peak_val < peak_thr:
        st.warning(f"当前峰值 {peak_val:.3f} 低于阈值 {peak_thr:g}，不触发联动生成（防误报）。")

    gen_clicked = st.button(
        "生成风浪联动与解读",
        type="primary",
        key="realtime_windwave_gen_btn",
        disabled=not can_gen,
    )
    if gen_clicked and last_rt and can_gen:
        try:
            defaults = infer_typhoon_link_defaults_from_eddy_result(last_rt)
            ar = build_anomaly_result_for_detect(last_rt, link_defaults=defaults)
            linked = run_detect(
                anomaly_result=ar,
                auto_link_typhoon=True,
                events_json_path=str(defaults["events_json_path"]),
                top_k=int(defaults["top_k"]),
            )
            st.session_state["realtime_windwave_linked"] = linked
            st.session_state["realtime_windwave_source_fp"] = _realtime_result_fingerprint(last_rt)
        except Exception as e:
            st.error(f"风浪联动失败：{e}")

    linked_saved = st.session_state.get("realtime_windwave_linked")
    cur_fp = _realtime_result_fingerprint(last_rt)
    if isinstance(linked_saved, dict) and linked_saved.get("typhoon_link") is not None:
        if st.session_state.get("realtime_windwave_source_fp") != cur_fp:
            st.info("以下联动基于此前一次快照生成；若实时结果已更新，请再次点击「生成风浪联动与解读」。")
        render_typhoon_linked_outputs(
            linked_saved,
            key_prefix="rt",
            kb_jump_params=None,
            llm_caption_extra="实时流：阈值达标后由用户手动触发生成。",
        )

    render_tri_module_strip(mode_label="实时系统 · 视频/RTSP")

