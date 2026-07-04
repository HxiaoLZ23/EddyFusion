# GOAL：风浪模块在线 LSTM 推理（替换平滑基线）

> **Goal ID**：`anomaly-lstm-online-inference`  
> **前置决策**：涡旋模块已锁定 **Fair-B0**（`config/eddy_v6_b0_fair.yaml`）；V6 Phase B0 goal 已 `done`。  
> **规范**：接手前阅读 `相关文件/AI_DEV_REQUIREMENTS.md`；增依赖同步 `requirements.txt`。

---

## 可直接粘贴的 `/goal` 提示词

```text
/goal 风浪在线 LSTM 推理：将 React/FastAPI 风浪链从「NC 平滑基线 pred」改为「双头 WindWaveLSTM 滑窗一步预测」，与训练/eval 口径一致，贯通异常分级、DTW 与前端曲线；不重训、不破坏 eval 指标链。

【背景】
- 当前在线链：`windwave_nc_bridge._smooth_baseline` → demo_wind/wave_predicted → eddy_typhoon_bridge._current_curve_for_detect → run_detect → DTW。
- 训练/eval 已有：`src/anomaly/model.py`（WindWaveLSTM in=2）、`outputs/anomaly/best.pt`、`config/anomaly.yaml`（window_hours=48, horizon_hours=1）。
- 离线滑窗推理参考：`scripts/anomaly_plot_residual_curves.py` 的 `_rolling_predict()`。
- 文档已标注缺口：`web_api/routers/windwave_report.py` 注释「预测侧当前为平滑基线」；`docs/工程手册/离线系统_预处理数据归档.md` §4.3。

【成功标准】
1. 新增可复用推理层（建议 `src/anomaly/inference.py` 或 `app/services/anomaly_inference_service.py`，参照 HydroInferenceService 编排模式）：加载 `config/anomaly.yaml` + `outputs/anomaly/best.pt`，输入 `(T,2)` 风/浪序列，输出与 obs 对齐的 `wind_predicted`/`wave_predicted` 列表（及 meta：ckpt、window_steps、device、fallback 原因）。
2. `windwave_nc_bridge.extract_wind_wave_companion_from_netcdf`（或紧邻封装）默认走 LSTM；`_smooth_baseline` 仅作显式降级（缺权重、T 过短、CUDA 不可用等），降级须在 `assessment_note` 或 response meta 中写明。
3. `POST /api/windwave/forecast`、`/offline-report`、async job `windwave_forecast` 返回的 pred 来自 LSTM；`current_curve`、DTW、异常色带与 LSTM pred 一致。
4. 单测：`tests/test_anomaly_inference.py`（mock 或小序列：pred 形状、降级路径）；扩展现有 `tests/test_anomaly_detect.py` 若需。
5. 文档：`docs/工程手册/离线系统_预处理数据归档.md` §4.3 更新为 LSTM 主路径 + 降级说明；路线图 G10 备注「在线已接 best.pt」；API docstring 去掉「平滑基线为主」误导。
6. 验收命令（须在本机或 CI 可跑）：`pytest tests/test_anomaly_detect.py tests/test_anomaly_inference.py -q`；可选 `python scripts/anomaly_plot_residual_curves.py --nc <demo_nc>` 与 API 同 ckpt 目视一致。

【非目标】
- 不重训 LSTM、不改 `src/anomaly/eval.py` 指标字段、不改 DTW/台风 KB 初筛逻辑、不改造水文、不恢复 7ch 涡旋 UI。
- 不把 Streamlit 作为主验收路径（React API 为主）。

【阶段】
S1 抽取 `_rolling_predict` 为 `src/anomaly/inference.py`（含 window/horizon 步长换算，默认 time_step_hours=3 与 preprocess 一致）。
S2 接入 `windwave_nc_bridge` + `build_eddy_result_from_windwave_netcdf`；保留 companion NPZ 演示路径。
S3 更新 `web_api/routers/windwave_report.py` meta 字段（prediction_backend: lstm|smooth_fallback）；job 进度文案改为准确表述。
S4 测试 + 文档 + 答辩脚本一句口径修正（`docs/开发规划/答辩演示脚本.md` §风浪）。

【不可更改】
- `config/anomaly.yaml` 的 level 含义、eval 输出字段、`run_eval_all` 企业材料指标链。
- 双头结构：in_features=2，wind_head/wave_head 输出 (B,2)。

【需人工】
- 若本机无 `outputs/anomaly/best.pt`，文档写明从 AutoDL 同步路径；代码用明确 FileNotFound 降级，不得静默假装 LSTM。
```

---

## Phase 清单（Agent 逐轮勾选）

| ID | 内容 | 验收 |
|----|------|------|
| S1 | `src/anomaly/inference.py`：`load_model`、`rolling_predict(series T×2)` | 单元测试通过 |
| S2 | `windwave_nc_bridge` 默认 LSTM pred | companion 四 list 来自 LSTM |
| S3 | API meta + job 文案 | forecast JSON 含 `prediction_backend` |
| S4 | pytest + 文档 §4.3 + 答辩脚本 | 命令绿、文档无「平滑为主」矛盾 |

---

## 关键文件索引

| 用途 | 路径 |
|------|------|
| 模型 | `src/anomaly/model.py` |
| 配置/权重 | `config/anomaly.yaml`，`outputs/anomaly/best.pt` |
| 现演示 pred | `src/anomaly/windwave_nc_bridge.py` |
| DTW 曲线 | `src/anomaly/eddy_typhoon_bridge.py` |
| API | `web_api/routers/windwave_report.py`，`web_api/job_queue.py` |
| 参考实现 | `scripts/anomaly_plot_residual_curves.py` |
| 水文编排参考 | `app/services/hydro_inference_service.py` |

---

## 禁止事项

1. 不得用 LSTM 替换后仍在前端/报告写「平滑基线」而不标注。  
2. 不得破坏 `outputs/anomaly/metrics_summary_*.json` 生成链。  
3. 不得在 `web_api` 内复制一整份 train 逻辑；推理层放 `src/anomaly/`。  
4. 涡旋交付线保持 **Fair-B0**，本 Goal 不碰 eddy 训练配置。
