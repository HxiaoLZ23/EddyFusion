from __future__ import annotations

import streamlit as st

from services.eddy_demo_service import EddyDemoService


def render() -> None:
    st.title("涡旋识别演示（真实推理）")
    st.caption("使用 YOLOv8-seg 权重对上传视频抽帧推理，展示检测时间轴与关键帧结果。")

    video_path = st.session_state.get("uploaded_video_path")
    if not video_path:
        st.warning("请先在“上传”页面上传视频。")
        return

    st.video(video_path)
    st.subheader("推理配置")
    c1, c2, c3 = st.columns(3)
    model_path = c1.text_input("权重路径", value="outputs/eddy/best.pt")
    conf = c2.slider("置信度阈值", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
    iou = c3.slider("IoU 阈值", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    max_frames = st.slider("最多检测帧数", min_value=20, max_value=300, value=120, step=10)
    st.caption("检测策略：对上传视频每 10 帧执行一次检测，并优先展示检测到涡旋的结果。")

    if st.button("运行涡旋真实推理", type="primary"):
        with st.spinner("正在推理并生成关键帧..."):
            svc = EddyDemoService(
                model_path=model_path,
                conf=conf,
                iou=iou,
                max_frames=max_frames,
                frame_stride=10,
            )
            try:
                out = svc.infer_video(video_path=video_path, task_id=st.session_state.get("task_id"))
                st.session_state["eddy_last_result"] = out
            except Exception as e:
                st.session_state["eddy_last_result"] = {"status": "failed", "message": str(e)}

    result = st.session_state.get("eddy_last_result")
    if not result:
        st.info("点击按钮开始推理。")
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
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("关键帧可视化")
    imgs = result.get("preview_images", [])
    if imgs:
        cols = st.columns(3)
        for i, p in enumerate(imgs):
            cols[i % 3].image(p, caption=f"frame_{i:02d}", use_container_width=True)
    else:
        st.caption("未生成关键帧图。")
