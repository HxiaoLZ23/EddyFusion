from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from services.hydro_inference_service import HydroInferenceService, HydroPreset
from services.nc_ingest_service import (
    MAX_NC_UPLOAD_MB,
    allowed_nc_suffixes_text,
    cleanup_old_nc_uploads,
    save_uploaded_nc,
)


def _render_path_check(check: dict[str, Any]) -> None:
    st.subheader("路径校验")
    c1, c2 = st.columns(2)
    c1.metric("配置文件", "存在" if check.get("config_exists") else "缺失")
    c2.metric("权重文件", "存在" if check.get("ckpt_exists") else "缺失")
    if check.get("messages"):
        for msg in check["messages"]:
            st.error(msg)
    else:
        st.success("路径校验通过")


def _render_result(result: dict[str, Any]) -> None:
    st.subheader("推理结果")
    warnings = result.get("warnings", [])
    if isinstance(warnings, list):
        for w in warnings:
            st.warning(str(w))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("状态", result.get("status", "unknown"))
    c2.metric("NRMSE(平均)", f"{float(result.get('nrmse_avg', 0.0)):.6f}")
    c3.metric("样本索引", str(result.get("sample_index", 0)))
    c4.metric("耗时(秒)", f"{float(result.get('elapsed_sec', 0.0)):.2f}")

    rmse_map = result.get("rmse_per_feature", {})
    nrmse_map = result.get("nrmse_per_feature", {})
    rows = []
    for feat in result.get("feature_names", []):
        rows.append(
            {
                "要素": feat,
                "RMSE": float(rmse_map.get(feat, 0.0)),
                "NRMSE": float(nrmse_map.get(feat, 0.0)),
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    feature_names = list(result.get("feature_names", []))
    if not feature_names:
        st.warning("结果中未包含可视化要素。")
        return
    selected_feature = st.selectbox("可视化要素", options=feature_names, index=0)
    if selected_feature:
        curve_rows = result["curve_data"][selected_feature]
        curve_data = {
            "gt": [float(r["gt"]) for r in curve_rows],
            "pred": [float(r["pred"]) for r in curve_rows],
        }
        st.markdown(f"**区域均值曲线（{selected_feature}）**")
        st.line_chart(curve_data, use_container_width=True)

        maps = result["map_data"][selected_feature]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(maps["gt"], cmap="viridis")
        axes[0].set_title("GT")
        axes[1].imshow(maps["pred"], cmap="viridis")
        axes[1].set_title("Pred")
        axes[2].imshow(maps["err"], cmap="magma")
        axes[2].set_title("|Err|")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{selected_feature} @ t+{result.get('t_last', '-')}")
        st.pyplot(fig, clear_figure=True)

    with st.expander("技术详情（可选）", expanded=False):
        st.json(
            {
                "split": result.get("split"),
                "dataset_size": result.get("dataset_size"),
                "level": result.get("level"),
                "device": result.get("device"),
                "sample_index": result.get("sample_index"),
                "rmse_per_feature": result.get("rmse_per_feature"),
                "nrmse_per_feature": result.get("nrmse_per_feature"),
            }
        )


def render(*, service: HydroInferenceService) -> None:
    st.title("水文推理演示（L2 优先）")
    st.caption("完整链路：模型配置 → NetCDF 缓冲与滑窗 → 推理 → 可视化；不再使用示例 NPZ 或上传 NPZ。")

    enable_l0 = st.toggle("启用 L0 实验开关（默认关闭）", value=False)
    presets = service.available_presets(enable_l0=enable_l0)
    labels = [p.label for p in presets]
    selected_label = st.selectbox("模型选择", options=labels, index=0)
    selected_preset: HydroPreset = next(p for p in presets if p.label == selected_label)

    c1, c2 = st.columns(2)
    with c1:
        config_path = st.text_input("配置路径", value=selected_preset.config_path)
    with c2:
        ckpt_path = st.text_input("权重路径", value=selected_preset.ckpt_path)
    sample_index = st.number_input("样本索引（对当前滑窗 NPZ 的窗口编号）", min_value=0, value=0, step=1)

    path_check = service.validate_paths(config_path=config_path, ckpt_path=ckpt_path)
    _render_path_check(path_check)

    st.subheader("输入：仅 NetCDF（会话缓冲 → 滑窗）")
    with st.expander("步长不足与实时接入：推荐策略（本页已实现缓冲 + FIFO）", expanded=False):
        st.markdown(
            """
**不要**用零填充或重复帧去「凑够」`input_steps+output_steps` 再跑已训练 ConvLSTM：会引入虚假动力学，指标与预警都不可信。

**推荐（与实时兼容）**

1. **步长不足时**：明确提示「当前 T 小于所需」，**不把未就绪数据当一次完整推理**；将已到达的文件路径**按时间顺序写入缓冲**（本页「追加到缓冲区」）。
2. **数据持续到达**：继续在缓冲尾部追加；当拼接后 **T ≥ input_steps+output_steps** 时，才允许「构建滑窗」并推理。
3. **缓冲上限**：实时流不能无限涨内存；采用 **按文件数 FIFO**（丢弃最旧日文件）。若将来服务端有单文件多时次，可改为「按时间步上限」裁切，原则相同：**丢最旧、保最新**。
4. **全部保留 vs 滑动窗口**：推理只需最近一段连续时间；**保留满足最长窗 + 少量余量即可**，不必永久保留全历史（归档应走对象存储/库表，而非推理缓冲）。

本页：**追加 → 显示估计 T 与所需 T → 够长再构建 NPZ → 推理**。
            """
        )

    st.caption(
        f"变量与 `config/variable_map.yaml` 对齐（{allowed_nc_suffixes_text()}，≤{MAX_NC_UPLOAD_MB}MB/文件）。"
        " 内部仍写出临时 NPZ 供 `HydroNpzDataset` 读取，你无需上传 NPZ。"
    )

    buf: list[str] = st.session_state.setdefault("hydro_nc_buffer", [])
    need_t = service.hydro_required_time_steps(config_path)
    try:
        est_t = service.peek_hydro_buffer_time_steps([Path(p) for p in buf], config_path=config_path) if buf else 0
    except Exception as e:
        est_t = -1
        peek_err = str(e)
    else:
        peek_err = None

    cbf1, cbf2 = st.columns(2)
    with cbf1:
        max_buf_files = st.number_input("缓冲区内最多保留日文件数（FIFO 丢最旧）", min_value=4, value=64, step=1)
    with cbf2:
        st.metric("所需连续时间步 T_need", str(need_t))
    m1, m2 = st.columns(2)
    m1.metric("缓冲文件数", str(len(buf)))
    if est_t >= 0:
        m2.metric("估计拼接时间步 T̂", str(est_t))
        if est_t < need_t:
            m2.caption("尚不足，请继续追加")
        else:
            m2.caption("可构建滑窗")
    else:
        m2.metric("估计 T̂", "—")
        if peek_err:
            m2.caption(peek_err[:80])
    if buf:
        with st.expander("缓冲区文件列表", expanded=False):
            for i, p in enumerate(buf):
                st.text(f"{i + 1}. {Path(p).name}")

    nc_multi = st.file_uploader(
        "选择本轮要追加的 NC（可多选，保存后按日期名排序参与拼接）",
        type=["nc", "nc4", "cdf"],
        accept_multiple_files=True,
        key="hydro_nc_multi",
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("追加到缓冲区", type="primary", key="hydro_buf_append"):
            if not nc_multi:
                st.warning("请先选择至少一个文件。")
            else:
                for f in nc_multi:
                    p, _ = save_uploaded_nc(f)
                    sp = str(p)
                    if sp not in buf:
                        buf.append(sp)
                cleanup_old_nc_uploads(max_files=60)
                mf = int(max_buf_files)
                while len(buf) > mf:
                    buf.pop(0)
                st.success(f"已追加，当前缓冲 {len(buf)} 个文件（上限 {mf}，超出则丢最旧）。")
    with b2:
        if st.button("清空缓冲区", key="hydro_buf_clear"):
            buf.clear()
            st.session_state.pop("hydro_nc_pair", None)
            st.session_state.pop("hydro_nc_build_meta", None)
            st.info("已清空。")

    nc_stride = st.number_input("滑窗 stride（越大窗口数 N 越少）", min_value=1, value=24, step=1)
    nc_cap = st.number_input("最多保留滑窗数（控内存）", min_value=1, value=256, step=1)

    if st.button("从缓冲区构建临时 X/y", type="primary", key="hydro_nc_materialize"):
        if not buf:
            st.warning("缓冲区为空，请先追加 NC。")
        else:
            try:
                xp, yp, meta = service.materialize_netcdf_to_xy_npz(
                    [Path(p) for p in buf],
                    config_path=config_path,
                    window_stride=int(nc_stride),
                    max_windows=int(nc_cap),
                )
                st.session_state["hydro_nc_pair"] = (str(xp), str(yp))
                st.session_state["hydro_nc_build_meta"] = meta
                st.success(
                    f"已生成 N={meta.get('n_windows')} 个滑窗；归一化: {meta.get('normalize', '')}。"
                )
                if meta.get("grid_warning"):
                    st.warning(meta["grid_warning"])
                if meta.get("normalize_note"):
                    st.info(meta["normalize_note"])
            except Exception as e:
                st.error(str(e))
                st.session_state.pop("hydro_nc_pair", None)
                st.session_state.pop("hydro_nc_build_meta", None)

    pair = st.session_state.get("hydro_nc_pair")
    bmeta = st.session_state.get("hydro_nc_build_meta")
    x_override: str | None = None
    y_override: str | None = None
    if pair and isinstance(pair, tuple) and len(pair) == 2:
        x_override, y_override = pair[0], pair[1]
        st.info("已就绪：可点击「运行水文推理」。")
        if isinstance(bmeta, dict) and bmeta:
            with st.expander("NC→滑窗元数据", expanded=False):
                st.json({k: v for k, v in bmeta.items() if k not in ("x_path", "y_path")})

    if st.button("运行水文推理", type="primary"):
        if not path_check.get("config_exists") or not path_check.get("ckpt_exists"):
            st.error("路径校验未通过，请先修复配置或权重路径。")
            return
        if not st.session_state.get("hydro_nc_pair"):
            st.error("请先「从缓冲区构建临时 X/y」（若 T 不足会报错，请继续追加 NC）。")
            return
        with st.spinner("正在执行模型推理..."):
            try:
                result = service.run(
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                    split="val",
                    sample_index=int(sample_index),
                    x_path_override=x_override,
                    y_path_override=y_override,
                    x_channel_indices=None,
                    y_channel_indices=None,
                )
                st.session_state["hydro_last_result"] = result
                st.session_state["hydro_last_preset"] = selected_preset.key
            except Exception as e:
                st.session_state["hydro_last_result"] = None
                st.error(f"推理失败：{e}")

    latest = st.session_state.get("hydro_last_result")
    if latest:
        _render_result(latest)
    else:
        st.info("尚未生成推理结果。")
