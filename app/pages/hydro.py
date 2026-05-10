from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from services.hydro_inference_service import HydroInferenceService, HydroPreset


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


def _render_metrics_summary(raw_items: list[dict[str, Any]]) -> None:
    st.subheader("离线评估摘要（metrics_summary）")
    for item in raw_items:
        path = item.get("path", "")
        exists = bool(item.get("exists"))
        with st.expander(path, expanded=False):
            if not exists:
                st.warning(f"不可用：{item.get('message', 'missing')}")
            else:
                st.success("读取成功")
                raw = item.get("raw", {})
                metrics = raw.get("metrics", {})
                if isinstance(metrics, dict) and metrics:
                    rows = []
                    for k, v in metrics.items():
                        if isinstance(v, dict):
                            for sk, sv in v.items():
                                rows.append({"指标": f"{k}.{sk}", "值": sv})
                        else:
                            rows.append({"指标": k, "值": v})
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                if st.toggle(f"显示 {path} 原始 JSON", key=f"raw_json_{path}"):
                    st.json(raw)


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
    st.caption("完整链路：模型配置 -> 输入 -> 推理 -> 图形化结果 -> 指标摘要")

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
    split = st.selectbox("评估数据划分", options=["val", "test"], index=0)
    sample_index = st.number_input("样本索引", min_value=0, value=0, step=1)

    path_check = service.validate_paths(config_path=config_path, ckpt_path=ckpt_path)
    _render_path_check(path_check)

    st.subheader("输入来源")
    input_mode = st.radio("输入模式", options=["示例数据（使用 config 内路径）", "上传 NPZ（X/y）"], index=0)
    x_override: str | None = None
    y_override: str | None = None
    x_channel_indices: list[int] | None = None
    y_channel_indices: list[int] | None = None
    if input_mode.startswith("上传"):
        up_x = st.file_uploader("上传 X 数据（.npz）", type=["npz"], key="hydro_x")
        up_y = st.file_uploader("上传 y 数据（.npz）", type=["npz"], key="hydro_y")
        if up_x is not None and up_y is not None:
            try:
                x_override = str(service.save_uploaded_npz(up_x))
                y_override = str(service.save_uploaded_npz(up_y))
                st.success("上传成功，已切换为上传数据推理。")
                req = service.feature_requirements(config_path=config_path)
                x_info = service.inspect_npz_channels(npz_path=x_override, preferred_key="X")
                y_info = service.inspect_npz_channels(npz_path=y_override, preferred_key="y")
                st.markdown("**通道映射（可选，解决通道不一致）**")
                st.caption(
                    f"模型期望 X={req['expected_in']} 通道({req['input_features']}), "
                    f"y={req['expected_out']} 通道({req['target_features']})；"
                    f"上传检测 X={x_info.get('channels', 0)} 通道, y={y_info.get('channels', 0)} 通道。"
                )
                x_opts = list(range(int(x_info.get("channels", 0))))
                y_opts = list(range(int(y_info.get("channels", 0))))
                x_default = x_opts[: int(req["expected_in"])] if len(x_opts) >= int(req["expected_in"]) else x_opts
                y_default = y_opts[: int(req["expected_out"])] if len(y_opts) >= int(req["expected_out"]) else y_opts
                x_sel = st.multiselect(
                    "X 通道映射顺序（映射到 input_features 顺序）",
                    options=x_opts,
                    default=x_default,
                    key="hydro_x_channel_map",
                )
                y_sel = st.multiselect(
                    "y 通道映射顺序（映射到 target_features 顺序）",
                    options=y_opts,
                    default=y_default,
                    key="hydro_y_channel_map",
                )
                if len(x_sel) == int(req["expected_in"]):
                    x_channel_indices = [int(i) for i in x_sel]
                elif x_sel:
                    st.warning(f"X 映射选择了 {len(x_sel)} 个通道，期望 {req['expected_in']} 个；将走自动截断/补零容错。")
                if len(y_sel) == int(req["expected_out"]):
                    y_channel_indices = [int(i) for i in y_sel]
                elif y_sel:
                    st.warning(f"y 映射选择了 {len(y_sel)} 个通道，期望 {req['expected_out']} 个；将走自动截断/补零容错。")
            except Exception as e:
                st.error(f"上传失败：{e}")

    if st.button("运行水文推理", type="primary"):
        if not path_check.get("config_exists") or not path_check.get("ckpt_exists"):
            st.error("路径校验未通过，请先修复配置或权重路径。")
            return
        with st.spinner("正在执行模型推理..."):
            try:
                result = service.run(
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                    split=split,
                    sample_index=int(sample_index),
                    x_path_override=x_override,
                    y_path_override=y_override,
                    x_channel_indices=x_channel_indices,
                    y_channel_indices=y_channel_indices,
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

    metrics_items = service.load_metrics_summary(selected_preset.metrics_paths)
    _render_metrics_summary(metrics_items)
