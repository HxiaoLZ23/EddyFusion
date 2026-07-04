from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st

from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path_for_stack
from services.nc_ingest_service import (
    MAX_NC_UPLOAD_MB,
    allowed_nc_suffixes_text,
    cleanup_old_nc_uploads,
    save_uploaded_nc,
)
from src.anomaly.eddy_typhoon_bridge import (
    apply_wind_wave_companion_to_eddy_result,
    strip_wind_wave_companion_from_eddy_result,
)
from src.anomaly.windwave_nc_bridge import (
    extract_wind_wave_companion_from_netcdf,
    wind_timeline_and_peak_from_companion,
)
from src.utils.config import resolve_path


def _strip_heavy_fields(payload: dict) -> dict:
    """避免把大图写入 eddy_last_result（风浪页 JSON 与 session 体积）。"""
    out = {k: v for k, v in payload.items() if k != "annotated_frame_bgr"}
    return out


def _resolve_nc_target(uploaded: object | None, rel: str) -> tuple[Path | None, str | None]:
    if uploaded is not None:
        try:
            p, _tid = save_uploaded_nc(uploaded)
            cleanup_old_nc_uploads(max_files=30)
            return p, None
        except Exception as e:
            return None, str(e)
    if rel.strip():
        p = resolve_path(rel.strip())
        if p.is_file():
            return p, None
        return None, f"路径无效或不是文件: {rel.strip()!r}"
    return None, "请上传 NC 或填写有效仓库相对路径。"


def render() -> None:
    st.title("涡旋识别（NetCDF → YOLO）")

    stack_label = st.radio(
        "YOLO 输入通道",
        (
            "3 通道（ADT+流场伪彩）",
            "7 通道（物理栈：`build_physics_stacked_hw7`，权重见 `outputs/eddy_enh7/`）",
        ),
        horizontal=True,
        key="eddy_yolo_stack_radio",
    )
    stack_id = "7ch" if stack_label.startswith("7") else "3ch"

    nc_mode = st.radio(
        "数据路径",
        options=("eddy_windwave", "eddy_only"),
        format_func=lambda x: {
            "eddy_windwave": "流场 + 风浪变量",
            "eddy_only": "仅流场",
        }[x],
        horizontal=False,
        key="eddy_nc_mode",
    )

    st.subheader("NetCDF")
    st.caption(f"{allowed_nc_suffixes_text()}，≤{MAX_NC_UPLOAD_MB}MB。")
    nc_u = st.file_uploader("上传 NetCDF", type=["nc", "nc4", "cdf"], key="eddy_page_nc_uploader")
    nc_rel = st.text_input(
        "或仓库相对路径（与上传二选一）",
        value="",
        key="eddy_nc_rel_path",
        placeholder="例: outputs/eddy_subset_19930101_20021231_small.nc",
    )
    nc_ti = st.number_input(
        "参考时次 time_index（风浪联动用）",
        min_value=0,
        value=24,
        step=1,
        key="eddy_nc_time_index",
        help="风浪页使用该时刻几何；视频仍按下方帧数上限逐帧推理。",
    )

    st.subheader("YOLO")
    model_path = st.text_input(
        "权重路径",
        value=default_eddy_weight_path_for_stack(stack_id),
        key=f"eddy_model_pt_{stack_id}",
        help="3ch → `outputs/eddy_v6_b0_fair/best.pt`（Fair-B0 默认）；7ch → `outputs/eddy_enh7/best.pt`（需与本页通道选项一致）。",
    )
    r1, r2 = st.columns(2)
    with r1:
        conf = st.slider("置信度 conf", min_value=0.05, max_value=0.9, value=0.25, step=0.05, key="eddy_nc_conf")
    with r2:
        iou = st.slider("IoU", min_value=0.1, max_value=0.9, value=0.45, step=0.05, key="eddy_nc_iou")
    base_imgsz = st.number_input("YOLO imgsz", min_value=320, max_value=1280, value=640, step=32)

    with st.expander("频域 / TTA（可选）", expanded=False):
        freq_mode = st.selectbox(
            "频域增强（推理前）",
            options=["none", "unsharp", "laplacian"],
            index=0,
        )
        freq_amt = st.slider("增强强度", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
        use_tta = st.checkbox("多尺度灵敏度", value=False)

    c_vid1, c_vid2 = st.columns(2)
    video_fps = c_vid1.slider("输出视频 FPS", min_value=1.0, max_value=24.0, value=2.0, step=1.0)
    video_max_frames = c_vid2.number_input(
        "视频最多推理帧数",
        min_value=8,
        max_value=300,
        value=120,
        step=8,
        help="长 NC 只取前 N 个 time_index；单时次 NC 会重复同一帧形成短视频。",
    )
    with st.expander("耗时 / 播放器说明", expanded=False):
        st.caption(
            "当前使用批量抽帧与批量 YOLO 推理，再编码 MP4。未安装 ffmpeg 时可能退回 mp4v，部分浏览器无法内嵌播放。"
        )

    if st.button("运行检测", type="primary", key="eddy_nc_run_all"):
        nc_target, err = _resolve_nc_target(nc_u, nc_rel)
        if err is not None:
            st.session_state["eddy_nc_result"] = {"status": "failed", "message": err}
            st.session_state.pop("eddy_last_result", None)
        else:
            tid = uuid.uuid4().hex[:12]
            st.session_state["task_id"] = tid
            st.session_state.pop("eddy_nc_mp4", None)
            svc = EddyDemoService(
                model_path=model_path,
                conf=conf,
                iou=iou,
                max_frames=1,
                frame_stride=1,
                base_imgsz=int(base_imgsz),
                frequency_mode=freq_mode,
                frequency_amount=float(freq_amt),
                multiscale_tta=use_tta,
            )
            try:
                with st.spinner("批量抽帧 + 批量 YOLO + 编码 MP4（加速模式）…"):
                    bundle = svc.infer_netcdf_dual_mp4(
                        nc_path=str(nc_target),
                        time_start=0,
                        time_stop=None,
                        time_stride=1,
                        fps=float(video_fps),
                        max_frames=int(video_max_frames),
                        task_id=tid,
                        deliver="full",
                    )
                if bundle.get("status") != "success":
                    st.session_state["eddy_nc_result"] = {
                        "status": "failed",
                        "message": str(bundle.get("message", bundle)),
                    }
                    st.session_state.pop("eddy_last_result", None)
                else:
                    # dual 路径返回视频信息；会话帧单独推一帧用于几何与后续风浪联动
                    used_indices = [int(x) for x in (bundle.get("time_indices") or [])]
                    if used_indices:
                        session_idx = min(used_indices, key=lambda x: abs(x - int(nc_ti)))
                    else:
                        session_idx = int(nc_ti)
                    raw = svc.infer_netcdf_frame(nc_path=str(nc_target), time_index=int(session_idx), task_id=None)
                    if raw.get("status") != "success":
                        raise RuntimeError(f"会话参考帧推理失败: {raw.get('message', raw)}")
                    ann_mp4 = bundle.get("annotated_mp4") or bundle.get("base_mp4")
                    mp4_path = str(resolve_path(str(ann_mp4))) if ann_mp4 else ""
                    merged_meta = dict(raw.get("meta") or {})
                    merged_meta.update(bundle.get("meta") or {})
                    merged_meta["mp4_path"] = mp4_path
                    merged_meta["video_n_frames"] = bundle.get("n_frames")
                    merged_meta["video_truncated"] = bundle.get("truncated")
                    merged_meta["session_time_index_used"] = int(session_idx)
                    st.session_state["eddy_nc_result"] = {
                        "status": "success",
                        "mp4_path": mp4_path,
                        "n_frames": bundle.get("n_frames"),
                        "truncated": bundle.get("truncated"),
                        "geometries": raw.get("geometries"),
                        "summary": raw.get("summary"),
                        "meta": merged_meta,
                        "task_id": raw.get("task_id"),
                    }
                    base = _strip_heavy_fields(dict(raw))

                    if nc_mode == "eddy_only":
                        st.session_state["_eddy_nc_wind_missing"] = False
                        merged = strip_wind_wave_companion_from_eddy_result(base)
                        merged["source_type"] = "netcdf_eddy_only"
                        merged["summary"] = "仅流场，风浪请在「风浪预警」单独上传 NC。"
                    else:
                        comp = extract_wind_wave_companion_from_netcdf(nc_target)
                        if comp is None:
                            st.session_state["_eddy_nc_wind_missing"] = True
                            merged = strip_wind_wave_companion_from_eddy_result(base)
                            merged["source_type"] = "netcdf_eddy_only"
                            merged["summary"] = "视频中未检测到风浪变量，风浪页按缺观测处理。"
                        else:
                            st.session_state["_eddy_nc_wind_missing"] = False
                            merged = apply_wind_wave_companion_to_eddy_result(base, comp)
                            w_tl, w_peak = wind_timeline_and_peak_from_companion(comp)
                            y_peak = float(merged.get("peak_score") or 0.0)
                            merged["timeline"] = w_tl
                            merged["peak_score"] = max(y_peak, w_peak)
                            merged["source_type"] = "netcdf_eddy_windwave"
                            merged["summary"] = "风浪时序已写入会话，可到「风浪预警」查看。"
                            em = dict(merged.get("meta") or {})
                            em["eddy_yolo_peak"] = y_peak
                            em["wind_combo_peak"] = w_peak
                            em["nc_path"] = str(nc_target)
                            merged["meta"] = em

                    st.session_state["eddy_last_result"] = merged
                    st.session_state.pop("last_result", None)
            except Exception as e:
                st.session_state["eddy_nc_result"] = {"status": "failed", "message": str(e)}
                st.session_state.pop("eddy_last_result", None)

    if st.session_state.pop("_eddy_nc_wind_missing", None):
        st.warning("当前 NC 无风浪变量，已按仅涡旋会话写入。")

    nc_page_res = st.session_state.get("eddy_nc_result")
    if nc_page_res and nc_page_res.get("status") == "success":
        st.success("涡旋检测完成（已生成视频）")
        vp = nc_page_res.get("mp4_path")
        if vp and Path(str(vp)).is_file():
            st.video(str(vp))
            meta = nc_page_res.get("meta") or {}
            if meta.get("video_encoding") == "mp4v_opencv":
                st.warning(meta.get("video_encoding_note") or "当前为 mp4v 编码，浏览器可能无法播放；请安装 ffmpeg 后重跑，或用 VLC 打开下载文件。")
            st.caption(
                f"共 {nc_page_res.get('n_frames', '?')} 帧；编码 {meta.get('video_encoding', '?')}；"
                f"截断={nc_page_res.get('truncated')}；`{vp}`"
            )
            with open(vp, "rb") as f:
                st.download_button(
                    "下载 MP4",
                    data=f.read(),
                    file_name=Path(vp).name,
                    mime="video/mp4",
                    key="eddy_nc_auto_vid_dl",
                )
        else:
            st.warning("未找到 MP4 文件，请重新运行检测。")
        if nc_page_res.get("meta"):
            with st.expander("NC / 推理元数据", expanded=False):
                st.json(nc_page_res["meta"])
        _show_geometries_panel(nc_page_res, key_suffix="nc")
    elif nc_page_res and nc_page_res.get("status") == "failed":
        st.error(str(nc_page_res.get("message")))

    sess = st.session_state.get("eddy_last_result")
    if isinstance(sess, dict) and sess.get("status") == "success":
        st.subheader("会话（风浪页读取）")
        c1, c2, c3 = st.columns(3)
        c1.metric("来源类型", str(sess.get("source_type", "-")))
        c2.metric("peak_score", f"{float(sess.get('peak_score', 0.0)):.4f}")
        c3.metric("时间轴长度", str(len(sess.get("timeline") or [])))


def _show_geometries_panel(result: dict, *, key_suffix: str) -> None:
    geoms = result.get("geometries") or []
    if not geoms:
        return
    with st.expander(f"实例几何属性（{len(geoms)} 条）", expanded=False):
        slim = []
        for g in geoms[:200]:
            slim.append(
                {
                    "inst": g.get("instance_id", g.get("frame_sample", "")),
                    "type": g.get("eddy_type"),
                    "area_px": round(float(g.get("area_pixels", 0)), 2),
                    "perimeter_px": round(float(g.get("perimeter_px", 0)), 2),
                    "angle°": g.get("angle_deg"),
                    "centroid": g.get("centroid_xy"),
                    "conf": round(float(g.get("confidence", 0)), 4),
                }
            )
        st.dataframe(slim, use_container_width=True, hide_index=True)
        st.download_button(
            "下载 geometries.json",
            data=json.dumps(geoms, ensure_ascii=False, indent=2),
            file_name=f"eddy_geometries_{key_suffix}.json",
            mime="application/json",
            key=f"dl_geo_{key_suffix}",
        )
