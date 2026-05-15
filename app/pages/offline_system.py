"""离线系统：以本地上传 NetCDF 为主数据源，布局与实时系统对齐（全链路占位）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from pages.ocean_tri_panel import render_tri_module_strip
from services.nc_ingest_service import (
    MAX_NC_UPLOAD_MB,
    allowed_nc_suffixes_text,
    cleanup_old_nc_uploads,
    save_uploaded_nc,
    summarize_nc_file,
)
from services.eddy_demo_service import EddyDemoService, default_eddy_weight_path
from services.nc_preprocess_facade import NcTaskBranch, describe_for_branch

_BRANCH_LABELS: dict[str, str] = {
    "eddy": "涡旋识别",
    "hydro": "水文预测",
    "windwave": "风浪预警",
    "full_chain": "全链路（占位）",
}


def render() -> None:
    st.title("离线系统")
    st.caption(
        f"手动上传 NetCDF（{allowed_nc_suffixes_text()}，单文件建议 ≤{MAX_NC_UPLOAD_MB}MB），"
        "统一预处理 Facade 摘要后驱动三模块（深度预处理与全链路推理建设中）。"
    )

    st.subheader("NetCDF 输入")
    files = st.file_uploader(
        "上传一个或多个 NC 文件",
        type=["nc", "nc4", "cdf"],
        accept_multiple_files=True,
        key="offline_nc_multi",
    )
    paths: list[Path] = []
    if files:
        for f in files:
            try:
                p, tid = save_uploaded_nc(f)
                paths.append(p)
                st.success(f"已保存：{f.name} → `{p.name}`（task {tid}）")
            except Exception as e:
                st.error(f"{f.name}：{e}")
        cleanup_old_nc_uploads(max_files=30)

    if paths:
        bkey = st.selectbox(
            "预处理目标分支（摘要用）",
            options=list(_BRANCH_LABELS.keys()),
            format_func=lambda k: _BRANCH_LABELS[k],
            key="offline_nc_branch",
        )
        branch = NcTaskBranch(bkey)
        if st.button("生成 NC 摘要（Facade）", key="offline_nc_describe"):
            summary: dict[str, Any] = describe_for_branch(paths, branch)
            st.session_state["offline_nc_summary"] = summary
        summ = st.session_state.get("offline_nc_summary")
        if isinstance(summ, dict):
            with st.expander("最近一次摘要", expanded=True):
                st.json(summ)
        for p in paths:
            s = summarize_nc_file(p)
            if s.get("error"):
                st.warning(f"{p.name}: {s['error']}")

        if branch == NcTaskBranch.EDDY:
            st.subheader("涡旋：NC → BGR → YOLO 单帧")
            st.caption(
                "从已上传的 NetCDF 抽取 ADT/流场或 SST/流场合成一帧，再调用与「涡旋识别」页相同的 YOLO 权重。"
                " 变量约定见 `src/eddy/nc_to_bgr.py`。"
            )
            labels = [p.name for p in paths]
            pick = st.selectbox("用于检测的 NC 文件", options=list(range(len(paths))), format_func=lambda i: labels[i])
            nc_pick = paths[int(pick)]
            ti = st.number_input("时间索引 time_index", min_value=0, value=24, step=1)
            eddy_model = st.text_input(
                "涡旋权重路径",
                value=default_eddy_weight_path(),
                key="offline_eddy_nc_model",
            )
            eddy_conf = st.slider("置信度 conf", min_value=0.05, max_value=0.9, value=0.25, step=0.05, key="offline_eddy_nc_conf")
            eddy_iou = st.slider("IoU", min_value=0.1, max_value=0.9, value=0.45, step=0.05, key="offline_eddy_nc_iou")
            eddy_imgsz = st.number_input("imgsz", min_value=320, max_value=1280, value=640, step=32, key="offline_eddy_nc_imgsz")
            if st.button("运行单帧涡旋检测", type="primary", key="offline_eddy_nc_run"):
                with st.spinner("NC 抽帧并推理…"):
                    svc = EddyDemoService(
                        model_path=eddy_model,
                        conf=float(eddy_conf),
                        iou=float(eddy_iou),
                        base_imgsz=int(eddy_imgsz),
                    )
                    try:
                        out = svc.infer_netcdf_frame(nc_path=str(nc_pick), time_index=int(ti), task_id=None)
                        st.session_state["offline_eddy_nc_result"] = out
                    except Exception as e:
                        st.session_state["offline_eddy_nc_result"] = {"status": "failed", "message": str(e)}

            nc_res = st.session_state.get("offline_eddy_nc_result")
            if isinstance(nc_res, dict):
                if nc_res.get("status") == "success":
                    st.success(str(nc_res.get("summary", "完成")))
                    meta = nc_res.get("meta") or {}
                    if meta:
                        with st.expander("NC / 推理元数据", expanded=False):
                            st.json(meta)
                    if nc_res.get("annotated_frame_bgr") is not None:
                        st.image(nc_res["annotated_frame_bgr"], channels="BGR", use_container_width=True)
                    geoms = nc_res.get("geometries") or []
                    if geoms:
                        with st.expander(f"实例几何（{len(geoms)}）", expanded=False):
                            st.json(geoms[:50])
                            if len(geoms) > 50:
                                st.caption("仅展示前 50 条；完整结果请在涡旋页导出。")
                elif nc_res.get("status") == "failed":
                    st.error(str(nc_res.get("message", "失败")))
    else:
        st.info("上传 NC 后可查看维度与变量列表摘要；完整推理请仍使用各「模块工作台」页面。")

    render_tri_module_strip(mode_label="离线系统 · 本地上传 NC")

    st.divider()
    st.markdown("**模块工作台**（单模块深链）")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("涡旋识别：调参、视频/NPZ、几何导出")
    with c2:
        st.caption("水文推理：L2 单样本与 NPZ 上传")
    with c3:
        st.caption("风浪预警：联动台风与 LLM 解读")
