# GOAL：风浪 LSTM 方案 A — Direct 多步预测（24h / 8×3h）

> **⚠️ 已 supersede**：完整业务链路（含 **Goal B 双曲线 DTW**）见 **[GOAL_风浪Direct24h与DTW链路升级.md](GOAL_风浪Direct24h与DTW链路升级.md)**。  
> **Goal ID**：`anomaly-direct-multistep-24h`（现作为子 Goal A）  
> **前置**：单步 3h 链已交付（`outputs/anomaly/best.pt`、`horizon_eval` 自回归 24h 仅作补充表）；本 Goal **必须重训**，旧权重不兼容。  
> **规范**：接手前阅读 `相关文件/AI_DEV_REQUIREMENTS.md`；增依赖同步 `requirements.txt`。

---

## 可直接粘贴的 `/goal` 提示词

```text
/goal 风浪 Direct 多步预测（方案 A）：将 WindWaveLSTM 从「48h 窗 → 单步 3h」改为「48h 窗 → 一次输出未来 8 步（24h，步长 3h）」；重预处理 npz、改模型 head、重训、重跑 eval/归档；贯通 inference 与在线 API；保留 3h 末步指标可对照，不破坏 eval JSON 字段约定。

【背景】
- 现状：`config/anomaly.yaml` → `horizon_hours: 1`（= 3h 一步）；`y` 形状 `(N,2)`；`WindWaveLSTM` 输出 `(B,2)`。
- 24h 现状：`src/anomaly/horizon_eval.py` 对单步权重做 8 步自回归 rollout，val mae_avg≈0.16（见 `submission/tables/anomaly_metrics_horizon24h.md`）。
- 目标：native direct multi-step（类似水文 `output_steps=72` 思路），默认 **horizon_hours=24**（8×3h）；可配置，但首版固定 24h。
- 数据：`config/data.yaml` → `anomaly_preprocess.time_step_hours: 3`；npz 由 `src/preprocess/anomaly_dataset.py` 生成。

【成功标准】
1. **配置**：`config/anomaly.yaml` 增加/改为 `horizon_hours: 24`（或等价 `horizon_steps: 8` 文档化）；meta 写入 npz。
2. **预处理**：`anomaly_dataset._build_windows` 产出 `y: (N, horizon_steps, 2)`（连续 H 步真值，非仅末点）；重跑 train/val/test npz；打印 shape 验收 `y=(?,8,2)`。
3. **模型**：`src/anomaly/model.py` — head 输出 `(B, horizon_steps, 2)` 或 `(B, 2*horizon_steps)`；`build_model` 从 cfg 读 horizon_steps；**保留** wind_head/wave_head 双头语义（可对每步共享权重或 per-step Linear，优先最小改动：两 Linear 各输出 H 维）。
4. **训练**：`src/anomaly/train.py` + `dataset.py` 对齐新 y；loss 为 H 步 MSE 均值（或末步+全步加权，默认全步均值）；产出新 `outputs/anomaly/best.pt`（或 `outputs/anomaly_multistep24/best.pt` + config 指向，二选一并在文档写明）。
5. **评估层 A**：
   - `src/anomaly/eval.py`：报告 `mae_avg`（全 H 步平均）、`mae_step3h`（第 1 步）、`mae_step24h`（第 8 步）；**保留** `mask_map50` 等字段不适用；JSON 仍含 `module/level/metrics/passed/tags`，`split` 不变。
   - `src/anomaly/horizon_eval.py`：direct 模式下改为「直接读第 H 步」或标记 `eval_mode: direct_native`，不再 rollout（或 rollout 仅作 ablation 对照）。
   - `src/anomaly/grid_eval.py`：若仍支持，对齐 H 步或仅评第 1 步并在 meta 说明。
6. **推理链**：`src/anomaly/inference.py` — `rolling_predict` 一次 forward 得 H 步；在线 pred 曲线用 **第 1 步** 与 obs 逐步对齐（滑窗）或文档说明展示策略；`windwave_nc_bridge` / `eddy_typhoon_bridge` 残差与 `current_curve` 与第 1 步 pred 一致（主链仍 3h 对齐观测网格）。
7. **测试**：`tests/test_anomaly_horizon_eval.py` 更新；新增 `tests/test_anomaly_multistep_shapes.py`（mock 小 batch：x (B,W,2), y (B,H,2), pred shape）。
8. **归档**：更新 `submission/tables/anomaly_metrics_val_test.md`；新增或更新 `anomaly_metrics_direct24h.md`；`docs/实验与结果归档/AutoDL_outputs_云端结果归档.md` §6.1/§6.5 注明「主链已改为 direct 24h 训练」与旧单步表对照；`docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md` 层 A 补一句多步口径。
9. **验收命令**（须跑通并贴关键数值）：
   - `python -m src.preprocess.anomaly_dataset`（或项目既有 preprocess 入口）
   - `python -m src.anomaly.train --config config/anomaly.yaml`
   - `python -m src.anomaly.eval --config config/anomaly.yaml`
   - `pytest tests/test_anomaly_horizon_eval.py tests/test_anomaly_multistep_shapes.py -q`
   - 对比：direct 24h 末步 MAE 应 ≤ 自回归 rollout 同口径（或文档解释未达标原因）

【非目标】
- 不改水文 ConvLSTM 72h、不改涡旋 Fair-B0、不改台风关联 Recall Oracle 评测、不改 DTW/KB 初筛逻辑。
- 不做 72h 风浪 direct（步数过多，首版仅 24h/8 步）。
- 不在本 Goal 内做云端 AutoDL 重训（本地/CI 烟测即可；云端路径写进「需人工」）。
- 不删除旧单步权重备份说明（可保留 `outputs/anomaly/best.pt` 改名为 `best_singlestep3h.pt.bak` 并在 README 注明）。

【阶段】
S1 配置 + `_build_windows` + npz 重生成（验收 y shape）。
S2 `model.py` + `dataset.py` + `train.py`（验收 1 epoch loss 下降）。
S3 全量 train → `best.pt`；`eval.py` / `horizon_eval.py` 多步指标。
S4 `inference.py` + bridge + API meta（`prediction_backend`, `horizon_hours`）。
S5 pytest + submission 表 + 归档文档 + 答辩口径一句（「训练为 24h direct，表 7-2 同时给 3h/24h 步 MAE」）。

【不可更改】
- `config/*.yaml` 的 `level` 含义；eval 输出顶层字段 `module/level/metrics/passed/tags`。
- `run_eval_all.sh` 若引用 anomaly eval，须仍能生成 JSON（字段可增不可删已有键）。
- 双任务结构：in_features=2，wind/wave 双头（不可合并为单标量头）。
- PyTorch≥2.5.1；路径相对仓库根。

【需人工】
- 全量重训若本地 GPU 不足：AutoDL 跑 `train` 后 sync `best.pt` 与 `metrics_summary_*.json`。
- 答辩/论文：勿将 direct 24h 末步 MAE 与旧单步 0.03 混在同一行不加脚注；台风层 B 仍弱讲 Oracle Recall。

【关键文件】
- `config/anomaly.yaml`, `config/data.yaml`
- `src/preprocess/anomaly_dataset.py`, `src/anomaly/model.py`, `src/anomaly/dataset.py`, `src/anomaly/train.py`, `src/anomaly/eval.py`, `src/anomaly/horizon_eval.py`, `src/anomaly/inference.py`, `src/anomaly/windwave_nc_bridge.py`
- `submission/tables/anomaly_metrics_*.md`, `docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md`
```

---

## Phase 清单（Agent 逐轮勾选）

| ID | 内容 | 验收 |
|----|------|------|
| S1 | `horizon_hours:24` + `_build_windows` → `y (N,8,2)` + 重生成 npz | `np.load` 打印 shape |
| S2 | 模型 head + train 1 epoch | loss finite、pred shape 正确 |
| S3 | 全量 train + eval + horizon_eval direct | 新 metrics JSON + 表 |
| S4 | inference + bridge + API meta | forecast 返回 H 步或文档化第 1 步对齐 |
| S5 | pytest + 归档 + 口径文档 | 验收命令全绿 |

---

## 建议 `/subgoal`（可选， mid-loop 追加）

```text
/subgoal eval JSON 必须同时输出 mae_step3h 与 mae_step24h，便于表 7-2 脚注

/subgoal 保留 horizon_eval 自回归模式作 --mode rollout 对照，不删旧脚本行为

/subgoal 新权重目录 outputs/anomaly_multistep24/ 与旧 outputs/anomaly/ 并存，config 可切换
```

---

## 答辩口径（完成后粘贴用）

> 风浪 LSTM 由 **3h 单步** 升级为 **24h direct 多步**（8×3h 一次输出）；主表同时报告 **3h 步 MAE** 与 **24h 末步 MAE**。24h 不再依赖单步自回归 rollout。台风关联仍为 Oracle 检索链验收，与 MAE 分层表述。
