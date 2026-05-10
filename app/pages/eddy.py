from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from services.eddy_demo_service import EddyDemoService


def render() -> None:
    st.title("涡旋识别演示（真实推理）")
    st.caption(
        "YOLOv8-seg 检测；支持 **频域增强预处理**、**多尺度灵敏度(TTA 提示)**、"
        "**SLA/涡度/温度梯度 NPZ→三通道融合** 与 **实例几何属性** 导出。"
    )

    video_path = st.session_state.get("uploaded_video_path")
    if not video_path:
        st.warning("请先在「上传」页面上传视频（视频链路与下方多通道单帧可并行使用）。")

    st.subheader("推理配置")
    c1, c2, c3 = st.columns(3)
    model_path = c1.text_input("权重路径", value="outputs/eddy/best.pt")
    conf = c2.slider("置信度阈值", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
    iou = c3.slider("IoU 阈值", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    max_frames = st.slider("最多检测帧数", min_value=20, max_value=300, value=120, step=10)
    base_imgsz = st.number_input("YOLO imgsz", min_value=320, max_value=1280, value=640, step=32)

    with st.expander("模块A 增强选项（频域 / 多尺度）", expanded=False):
        freq_mode = st.selectbox(
            "频域增强（推理前）",
            options=["none", "unsharp", "laplacian"],
            index=0,
            help="可视化与轻量消融：unsharp/laplacian 突出边界；非训练内置频域支路。",
        )
        freq_amt = st.slider("增强强度", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
        use_tta = st.checkbox(
            "多尺度灵敏度",
            value=False,
            help="在多个 imgsz 上任一出现框则 timeline 记 tta_any_hit；出图与几何仍用主尺度。",
        )

    st.subheader("多通道物理场（单帧 NPZ）")
    st.caption("NPZ 需含可识别键：sla/adt/ssh、vorticity/vor、temp_grad/dtdy 等组合，见 `src/eddy/multichannel_fuse.py`。")
    npz_file = st.file_uploader("上传融合用 NPZ（可选）", type=["npz"])
    if npz_file is not None and st.button("对 NPZ 融合场运行单帧检测", key="run_npz_mc"):
        tmp = Path("app/data/eddy_preview") / f"_upload_{npz_file.name}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(npz_file.getvalue())
        svc = EddyDemoService(
            model_path=model_path,
            conf=conf,
            iou=iou,
            max_frames=max_frames,
            frame_stride=10,
            base_imgsz=int(base_imgsz),
            frequency_mode=freq_mode,
            frequency_amount=float(freq_amt),
            multiscale_tta=use_tta,
        )
        try:
            out = svc.infer_multichannel_npz(npz_path=str(tmp), task_id=st.session_state.get("task_id"))
            st.session_state["eddy_npz_result"] = out
        except Exception as e:
            st.session_state["eddy_npz_result"] = {"status": "failed", "message": str(e)}

    npz_res = st.session_state.get("eddy_npz_result")
    if npz_res and npz_res.get("status") == "success":
        st.success("NPZ 融合场检测完成")
        if npz_res.get("annotated_frame_bgr") is not None:
            st.image(npz_res["annotated_frame_bgr"], channels="BGR", use_container_width=True)
        _show_geometries_panel(npz_res, key_suffix="npz")
    elif npz_res and npz_res.get("status") == "failed":
        st.error(str(npz_res.get("message")))

    if video_path:
        st.video(video_path)
        st.caption("检测策略：按 stride 抽帧；可选频域增强与多尺度灵敏度见上方。")

    stride = st.slider("抽帧间隔", min_value=1, max_value=30, value=10, step=1)

    if video_path and st.button("运行涡旋真实推理", type="primary"):
        with st.spinner("正在推理并生成关键帧..."):
            svc = EddyDemoService(
                model_path=model_path,
                conf=conf,
                iou=iou,
                max_frames=max_frames,
                frame_stride=stride,
                base_imgsz=int(base_imgsz),
                frequency_mode=freq_mode,
                frequency_amount=float(freq_amt),
                multiscale_tta=use_tta,
            )
            try:
                out = svc.infer_video(video_path=video_path, task_id=st.session_state.get("task_id"))
                st.session_state["eddy_last_result"] = out
            except Exception as e:
                st.session_state["eddy_last_result"] = {"status": "failed", "message": str(e)}

    result = st.session_state.get("eddy_last_result")
    if not result:
        if video_path:
            st.info("点击按钮开始视频推理。")
        return
    if result.get("status") != "success":
        st.error(str(result.get("message", "推理失败")))
        return

    st.subheader("推理结果")
    r1, r2, r3 = st.columns(3)
    r1.metric("状态", str(result.get("status", "-")))
    r2.metric("峰值分数", f"{float(result.get('peak_score', 0.0)):.3f}")
    r3.metric("抽帧数", str(result.get("meta", {}).get("sampled_frames", 0)))
    if result.get("summary"):
        st.success(str(result.get("summary")))
    for w in result.get("warnings", []) if isinstance(result.get("warnings"), list) else []:
        if w:
            st.warning(str(w))

    timeline = result.get("timeline", [])
    if timeline:
        rows = []
        for it in timeline:
            rows.append(
                {
                    "时间": it.get("time", "-"),
                    "事件": it.get("event", "-"),
                    "数量": it.get("count", 0),
                    "分数": round(float(it.get("score", 0.0)), 4),
                    "TTA任一命中": it.get("tta_any_hit", ""),
                    "实例数": it.get("instances", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    _show_geometries_panel(result, key_suffix="vid")

    st.subheader("关键帧可视化")
    imgs = result.get("preview_images", [])
    if imgs:
        cols = st.columns(3)
        for i, p in enumerate(imgs):
            cols[i % 3].image(str(p), caption=f"frame_{i:02d}", use_container_width=True)
    else:
        st.caption("未生成关键帧图。")


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

