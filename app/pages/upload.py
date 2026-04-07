from __future__ import annotations

import streamlit as st

from services.media_service import (
    MAX_UPLOAD_MB,
    allowed_suffixes_text,
    cleanup_old_media,
    extract_video_metadata,
    save_uploaded_video,
)


def render() -> None:
    st.title("视频上传与预览")
    st.caption(f"支持格式：{allowed_suffixes_text()}，单文件最大建议 {MAX_UPLOAD_MB}MB")

    file = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi", "mkv", "webm"])
    if file is None:
        if st.session_state.get("uploaded_video_path"):
            st.info(f"当前会话已上传：{st.session_state.get('uploaded_video_name')}")
        return

    try:
        saved_path, task_id = save_uploaded_video(file)
        cleanup_old_media(max_files=20)
        metadata = extract_video_metadata(saved_path)
    except Exception as e:
        st.error(f"上传或解析失败：{e}")
        return

    st.session_state["task_id"] = task_id
    st.session_state["uploaded_video_path"] = str(saved_path)
    st.session_state["uploaded_video_name"] = file.name
    st.session_state["uploaded_video_meta"] = metadata
    st.session_state["task_status"] = "uploaded"

    st.success(f"上传成功，任务ID：{task_id}")
    st.video(str(saved_path))

    st.subheader("媒体元信息")
    c1, c2, c3 = st.columns(3)
    c1.metric("文件大小(MB)", f"{metadata.get('size_mb', 0.0):.2f}")
    c2.metric("分辨率", metadata.get("resolution", "unknown"))
    c3.metric("时长(秒)", str(metadata.get("duration_sec", "unknown")))
    st.json(metadata)

