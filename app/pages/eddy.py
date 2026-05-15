from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st

from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path
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
    st.title("涡旋识别（NetCDF 格点）")
    st.caption(
        "本页**仅支持 NetCDF**：点击「运行检测」后，对时间维逐帧推理并**自动生成带识别框的 MP4** 播放；"
        "「会话参考时次」用于风浪联动与几何列表，与视频中对应帧一致。"
        " 与风浪页 NC 入口共用 `eddy_last_result`，以最后一次成功写入为准。"
    )

    nc_mode = st.radio(
        "数据路径",
        options=("eddy_windwave", "eddy_only"),
        format_func=lambda x: {
            "eddy_windwave": "① 上传含流场 + 风浪要素的 NC → 涡旋 + 风浪会话",
            "eddy_only": "② 上传仅有流场（无风浪变量）的 NC → 仅涡旋结果",
        }[x],
        horizontal=False,
        key="eddy_nc_mode",
    )

    st.subheader("NetCDF")
    st.caption(
        f"{allowed_nc_suffixes_text()}，单文件 ≤{MAX_NC_UPLOAD_MB}MB。"
        " 流场用于 `src/eddy/nc_to_bgr.py`（ADT/UGOS/VGOS 或 SST/SSU/SSV 等）；"
        " 风浪用于 `extract_wind_wave_companion_from_netcdf`（u10/v10、有效波高等）。"
    )
    nc_u = st.file_uploader("上传 NetCDF", type=["nc", "nc4", "cdf"], key="eddy_page_nc_uploader")
    nc_rel = st.text_input(
        "或仓库相对路径（与上传二选一）",
        value="",
        key="eddy_nc_rel_path",
        placeholder="例: outputs/eddy_subset_19930101_20021231_small.nc",
    )
    nc_ti = st.number_input(
        "会话参考时次 time_index",
        min_value=0,
        value=24,
        step=1,
        key="eddy_nc_time_index",
        help="风浪会话与下方几何列表使用该时刻的检测结果；视频仍覆盖 NC 前 min(T, 上限) 个时刻。越界自动钳位。",
    )

    st.subheader("YOLO 配置")
    c1, c2, c3 = st.columns(3)
    model_path = c1.text_input(
        "权重路径",
        value=default_eddy_weight_path(),
        help="若存在 eddy_enh 的 best.pt 则默认选用；NetCDF 会自动走 8ch 物理堆叠或 3ch 伪彩。",
    )
    conf = c2.slider("置信度阈值", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
    iou = c3.slider("IoU 阈值", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    base_imgsz = st.number_input("YOLO imgsz", min_value=320, max_value=1280, value=640, step=32)

    with st.expander("频域 / 多尺度（可选）", expanded=False):
        freq_mode = st.selectbox(
            "频域增强（推理前）",
            options=["none", "unsharp", "laplacian"],
            index=0,
        )
        freq_amt = st.slider("增强强度", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
        use_tta = st.checkbox("多尺度灵敏度", value=False)

    c_vid1, c_vid2, c_vid3 = st.columns(3)
    video_fps = c_vid1.slider("输出视频 FPS", min_value=1.0, max_value=24.0, value=2.0, step=1.0)
    video_max_frames = c_vid2.number_input(
        "视频最多推理帧数",
        min_value=8,
        max_value=300,
        value=120,
        step=8,
        help="长 NC 只取前 N 个 time_index；单时次 NC 会重复同一帧形成短视频。",
    )
    single_repeat = c_vid3.number_input(
        "单时次 NC 重复帧数",
        min_value=8,
        max_value=120,
        value=36,
        step=4,
        help="无时间维或 T=1 时，将唯一一帧复制多次再编码。",
    )

    with st.expander("为何较慢 / 浏览器无法播放 MP4？", expanded=False):
        st.markdown(
            """
**耗时主要来自：**

1. **按帧重复整条推理链**：每个 `time_index` 都会重新打开 NetCDF、抽场、（8ch 时）算物理堆叠，再跑 **YOLO**；帧数 × 单帧耗时近似线性增长。  
2. **大格点**：命题方全图（如数百×上千）比裁剪子集慢得多，显存/CPU 压力也大。  
3. **设备**：未用 GPU 或 batch=1 时，推理明显变慢。  
4. **编码**：已优先用 **ffmpeg → H.264（yuv420p）**；若系统未装 ffmpeg 会退回 OpenCV **mp4v**，文件体积小但**多数浏览器不能内嵌播放**。

**建议：** 用 `python scripts/extract_eddy_subset_nc.py` 从大 NC 截一小片 + 少量时间步；适当降低「视频最多推理帧数」；Windows 可安装 [ffmpeg](https://ffmpeg.org/download.html) 并加入 PATH。
            """.strip()
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
                with st.spinner("逐帧涡旋推理并编码 MP4（可能需数十秒）…"):
                    bundle = svc.infer_netcdf_detection_video(
                        nc_path=str(nc_target),
                        session_time_index=int(nc_ti),
                        fps=float(video_fps),
                        max_frames=int(video_max_frames),
                        single_time_repeats=int(single_repeat),
                        task_id=tid,
                    )
                if bundle.get("status") != "success":
                    st.session_state["eddy_nc_result"] = {
                        "status": "failed",
                        "message": str(bundle.get("message", bundle)),
                    }
                    st.session_state.pop("eddy_last_result", None)
                else:
                    raw = bundle["session_frame"]
                    if not isinstance(raw, dict):
                        raise TypeError("session_frame 格式异常")
                    mp4_path = str(bundle["mp4_path"])
                    merged_meta = dict(raw.get("meta") or {})
                    merged_meta.update(bundle.get("meta") or {})
                    merged_meta["mp4_path"] = mp4_path
                    merged_meta["video_n_frames"] = bundle.get("n_frames")
                    merged_meta["video_truncated"] = bundle.get("truncated")
                    merged_meta["session_time_index_used"] = bundle.get("session_time_index_used")
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
                        merged["summary"] = "NC：仅涡旋（路径②）；已生成检测视频，未写入风浪 demo 序列。"
                    else:
                        comp = extract_wind_wave_companion_from_netcdf(nc_target)
                        if comp is None:
                            st.session_state["_eddy_nc_wind_missing"] = True
                            merged = strip_wind_wave_companion_from_eddy_result(base)
                            merged["source_type"] = "netcdf_eddy_only"
                            merged["summary"] = (
                                "NC：已生成检测视频；该文件**未识别到风浪变量**（路径① 未满足 u10/v10 或浪高等），"
                                "风浪页将按缺观测/演示代理处理。"
                            )
                        else:
                            st.session_state["_eddy_nc_wind_missing"] = False
                            merged = apply_wind_wave_companion_to_eddy_result(base, comp)
                            w_tl, w_peak = wind_timeline_and_peak_from_companion(comp)
                            y_peak = float(merged.get("peak_score") or 0.0)
                            merged["timeline"] = w_tl
                            merged["peak_score"] = max(y_peak, w_peak)
                            merged["source_type"] = "netcdf_eddy_windwave"
                            merged["summary"] = (
                                "NC：检测视频已生成 + 风浪时序已写入会话，可打开「风浪预警」查看 `run_detect` 与台风联动。"
                            )
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
        st.warning("路径① 需要风浪变量：当前 NC 未通过提取，已退回为**仅涡旋**会话。")

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
        st.subheader("会话摘要（供风浪预警页）")
        st.caption("`eddy_last_result` 不含大图；打开「风浪预警」前请勿混淆与「风浪页独立上传 NC」的覆盖关系。")
        c1, c2, c3 = st.columns(3)
        c1.metric("来源类型", str(sess.get("source_type", "-")))
        c2.metric("peak_score", f"{float(sess.get('peak_score', 0.0)):.4f}")
        c3.metric("时间轴长度", str(len(sess.get("timeline") or [])))
        if sess.get("wind_wave_from_companion_npz"):
            st.info("已含风浪 demo 序列，风浪页可直接联动。")
        if sess.get("summary"):
            st.info(str(sess.get("summary")))


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
                    "area_px": round(float(g.get("area_pixels", 0)), 2),
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
