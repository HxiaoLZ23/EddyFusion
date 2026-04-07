from __future__ import annotations

import streamlit as st

from services.inference_service import InferenceInput, InferenceService


def render(*, inference_service: InferenceService) -> None:
    st.title("结果展示")
    video_path = st.session_state.get("uploaded_video_path")
    if not video_path:
        st.warning("请先在“上传”页面上传视频。")
        return

    st.caption(f"任务ID: {st.session_state.get('task_id', 'N/A')}")
    st.video(video_path)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("运行演示推理（Mock）", type="primary"):
            st.session_state["task_status"] = "processing"
            try:
                result = inference_service.run(
                    InferenceInput(
                        source_type="upload",
                        task_id=st.session_state.get("task_id"),
                        video_path=video_path,
                        timestamp_ms=None,
                        metadata={"uploaded_name": st.session_state.get("uploaded_video_name")},
                    )
                )
                st.session_state["last_result"] = result
                st.session_state["task_status"] = "done"
            except Exception as e:
                st.session_state["task_status"] = "failed"
                st.session_state["last_result"] = {
                    "task_id": st.session_state.get("task_id"),
                    "status": "failed",
                    "message": str(e),
                    "timeline": [],
                }
                st.error(f"推理失败：{e}")

    with c2:
        st.info("真实推理适配层已预留，当前默认 Mock。")

    st.subheader("任务状态")
    st.write(st.session_state.get("task_status", "idle"))

    result = st.session_state.get("last_result")
    if not result:
        st.caption("尚无结果，点击按钮开始。")
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

    with st.expander("查看原始结果 JSON", expanded=False):
        st.json(result)

