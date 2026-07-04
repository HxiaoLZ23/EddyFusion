# GOAL：风浪业务链路重构 — Direct 24h 预测 + 双曲线 DTW

> **Goal ID（总）**：`anomaly-pipeline-direct24h-dtw-v2`  
> **子 Goal**：A `anomaly-direct-multistep-24h` · B `anomaly-dtw-forecast-curve`  
> **定位**：**业务链路重构**，非单纯改 head；模型升级必须与 DTW/异常/答辩叙事同步，避免「24h 预测 + 3h 残差 DTW」逻辑断裂。  
> **规范**：接手前阅读 `相关文件/AI_DEV_REQUIREMENTS.md`；增依赖同步 `requirements.txt`。

---

## 为什么要重写（旧 Goal 的隐含问题）

旧 Goal 写「Direct 24h + inference + **残差/current_curve 与第 1 步 pred 一致**」，等于 **DTW 仍绑定 step1 残差**，24h 预测不参与历史事件关联。

答辩必问：

> 你为什么做 24h 预测？—— 为了提前预警。  
> 那 DTW 为什么还看 3h 残差？—— **答不上来。**

**更合理的业务故事（论文 §5 对齐）**：

```text
WindWaveLSTM
    ↓
未来 24h Direct 预测（forecast）
    ↓
├── 异常检测链：pred_step1 vs obs → 3σ 分级（detection_curve）
└── 历史事件关联：forecast_curve → KB 时空初筛 → DTW 重排（forecast_curve）
         ↓
    风险评估 + Top-K 历史类比 + 报告
```

---

## 一键粘贴：总 Goal（推荐）

```text
/goal 风浪业务链路重构 v2：Goal A Direct24h 多步预测 + Goal B 双曲线 DTW 升级；模型、inference、detect、API、前端、评测与论文叙事一致；禁止「24h 预测仍用 step1 残差做 DTW」的旧口径。

【论文叙事（必须贯穿实现与文档）】
LSTM → 未来24h预测 → 风险评估（3σ 仍基于逐步对齐的 step1 残差）→ DTW 历史事件关联（基于 forecast_curve，非 past residual）。
定位：风浪异常分析与**预警**；24h 为预警超前，DTW 解释「未来态势像哪场历史台风」。

【现状与缺口】
- 模型：`horizon_hours=1`，`y:(N,2)`，`WindWaveLSTM`→`(B,2)`；权重 `outputs/anomaly/best.pt`。
- DTW：`link_anomaly_to_typhoon` 仅读 `anomaly_result.current_curve`；`eddy_typhoon_bridge._current_curve_for_detect` = |obs−pred| 全历史（`src/anomaly/detect.py` `rerank_candidates_by_dtw`）。
- 24h 评估：仅 MAE rollout（`horizon_eval.py`），无 forecast-DTW 指标占位。
- 文档：`风浪异常_指标口径与台风关联评测.md` 仍写 residual→DTW 单路径。

【Goal A — Direct 24h 多步预测】
A1. `config/anomaly.yaml`：`horizon_hours: 24`（8×3h）；npz meta 写入 horizon_steps。
A2. `anomaly_dataset._build_windows` → `y:(N,H,2)`；重生成 train/val/test npz。
A3. `model.py` 双头输出 `(B,H,2)`；`train.py`/`dataset.py` 对齐；H 步 MSE 均值；新权重 `outputs/anomaly_multistep24/best.pt`（与旧单步并存，config 可切换）。
A4. `eval.py`：`mae_avg`、`mae_step3h`、`mae_step24h`；JSON 顶层字段不删。
A5. `horizon_eval.py`：`eval_mode: direct_native`；保留 `--mode rollout` 对照。

【Goal B — 双曲线 DTW 链路（S4.5，与 A 同批交付）】
B1. **曲线语义拆分（代码字段名固定）**：
   - `detection_curve`（别名兼容 `current_curve`）：|wind_obs−wind_pred_step1| + |wave_obs−wave_pred_step1|，与 obs 时间轴对齐；供 `compute_anomaly_assessment`（3σ）与异常色带。
   - `forecast_curve`：来自 **当前时刻一次 forward 的 H 步未来预测**，长度 H（或 8）；定义 `forecast_curve[i] = pred_wind[i] + pred_wave[i]`（H 步，无 obs 相减）；供 `link_anomaly_to_typhoon` **DTW 重排**。
B2. `src/anomaly/detect.py`：
   - `rerank_candidates_by_dtw(..., query_curve=...)` 泛化；`link_anomaly_to_typhoon` 优先 `forecast_curve`，缺失时降级 `detection_curve` 并 meta 标注 `dtw_curve_source: detection_fallback`。
   - `run_detect`：3σ 仍用 step1 残差字段；DTW 默认 `forecast_curve`。
B3. `src/anomaly/inference.py`：
   - `predict_multistep(series)` → `{pred_step1, pred_future:(H,2), forecast_curve:(H,)}`；
   - 滑窗在线：`wind_predicted`/`wave_predicted` 列表仍与 obs **逐步对齐**（每步用 step1）；另返回 `forecast_future_wind/wave` 或 `forecast_bundle` 供 DTW/API。
B4. `eddy_typhoon_bridge.py` / `windwave_nc_bridge.py`：
   - 拆分 `_detection_curve_for_assess` 与 `_forecast_curve_for_dtw`；`build_anomaly_result_for_detect` 同时写入两曲线 + `wind_predicted`/`wind_observed`（step1 对齐）。
B5. **API**（`web_api/routers/windwave_report.py` 等）：
   - 响应 meta：`prediction_horizon_hours: 24`，`dtw_curve_source: forecast|detection_fallback`；
   - JSON 含 `detection_curve`、`forecast_curve`（或 `forecast_curve_wind/wave`）；Top-K 候选带 `dtw_distance`（基于 forecast）。
B6. **前端**（`web/` 风浪页，最小改动）：
   - 曲线区：obs + step1 pred + 异常色带（检测链）；
   - Top-K 卡片脚注：「DTW 基于未来24h预测曲线」；可选展示 forecast 小曲线。
B7. **评测扩展**（本版可占位 + 烟测，不必 Oracle Recall=1 复现）：
   - 新增 `scripts/anomaly_forecast_dtw_eval.py` 或扩展 `anomaly_typhoon_link_eval.py`：`--dtw-curve forecast`；
   - 输出 JSON 预留字段：`forecast_dtw_enabled`、`forecast_top1_event_id`；完整 `forecast_topk_hit` / `forecast_dtw_score` 可标 `status: placeholder`。
   - `horizon_eval` 产物 meta 注明：MAE 评预测；DTW 评关联，二者分层。

【成功标准（验收勾选）】
1. npz `y.shape[-2]==8`；train 1 epoch + 全量 train 可跑。
2. inference 单次 forward 返回 H 步 + `forecast_curve` 长度 H。
3. `run_detect` 在有多步 pred 时 DTW 使用 `forecast_curve`（单测 assert `retrieval.dtw.curve_source==forecast`）。
4. 3σ 等级仍由 step1 残差驱动（单测：仅改 forecast 不改 step1 → level 不变）。
5. API `/forecast` 返回两曲线 + meta；pytest 绿。
6. 文档：`风浪异常_指标口径与台风关联评测.md` 新增 **§Direct24h 双曲线 DTW 链路**（旧 residual→DTW 标为 legacy）；`离线系统_预处理数据归档.md` §4.3 更新；`答辩演示脚本.md` 风浪段改口播。
7. submission：`anomaly_metrics_direct24h.md` + 可选 `anomaly_forecast_dtw_eval.json` 占位。

【非目标】
- 不改水文 72h、涡旋 Fair-B0；不删 Oracle 时空 Recall 脚本（层 B 对照保留）。
- 不做 72h 风浪 direct；不要求本 Goal 内云端全量重训（写需人工）。
- 不要求端到端「格点 3σ 自动框台风」POD。

【阶段顺序】
S1 Goal A：配置+npz+model+train 烟测
S2 Goal A：全量 train + eval + horizon_eval direct
S3 Goal B：detect/inference 双曲线 + bridge
S4 Goal B：API + 前端 meta/脚注
S4.5 Goal B：forecast DTW eval 占位脚本 + 单测
S5 文档+submission+答辩口径统一

【不可更改】
config `level` 含义；eval JSON 顶层 `module/level/metrics/passed/tags`；PyTorch≥2.5.1；路径相对仓库根；`current_curve` 作 detection 别名保留兼容旧客户端。

【需人工】
GPU 全量重训 sync；论文 §5 图更新为双曲线示意；层 B Oracle Recall 仍弱讲，与 forecast DTW 分层表述。

【验收命令】
python -m src.preprocess.anomaly_dataset
python -m src.anomaly.train --config config/anomaly.yaml
python -m src.anomaly.eval --config config/anomaly.yaml
pytest tests/test_anomaly_multistep_shapes.py tests/test_anomaly_detect.py tests/test_anomaly_forecast_dtw.py -q
python scripts/anomaly_typhoon_link_eval.py --help  # 确认新 --dtw-curve 存在或占位脚本可跑
```

---

## 分 Goal 粘贴（可拆两次 /goal）

### Goal A 仅模型与 MAE

```text
/goal 风浪 Goal A：WindWaveLSTM 单步3h → Direct24h（8×3h）；重预处理/重训/eval；产出 pred_future (H,2)；为 Goal B forecast_curve 提供张量来源。不修改 DTW 曲线选择（由 Goal B 接 forecast_curve）。

成功标准：y(N,8,2)；model(B,8,2)；mae_step3h/mae_step24h；weights outputs/anomaly_multistep24/best.pt。
非目标：不改 detect DTW 默认曲线、不改 API 字段。
```

### Goal B 仅 DTW 业务链

```text
/goal 风浪 Goal B：双曲线 DTW 链路升级——detection_curve=|obs-pred_step1|（3σ+色带）；forecast_curve=pred_wind+pred_wave 未来H步（DTW+Top-K）；link_anomaly_to_typhoon 默认 forecast；API/文档/答辩口径同步。依赖 Goal A 的 pred_future 或 mock (H,2) 烟测。

成功标准：run_detect meta dtw_curve_source=forecast；API 返回两曲线；文档 §Direct24h 双曲线；单测 test_anomaly_forecast_dtw.py。
非目标：不重训 LSTM（可 mock 多步输出）；不要求 forecast_topk_hit 全量 IBTrACS 数值。
```

---

## 数据流与字段契约

```text
inference.predict_at_t(now):
  pred_step1: (2,)           # wind, wave — 与 obs[t+1] 对齐
  pred_future: (H, 2)       # 未来 H 步 direct
  forecast_curve: (H,)      # pred_future[:,0] + pred_future[:,1]

build_anomaly_result_for_detect:
  wind_observed / wind_predicted   # 逐步滑窗，step1 对齐
  wind_residual / wave_residual    # 当前时刻 step1
  detection_curve                  # 全序列 |obs-pred_step1| 组合（兼容 current_curve）
  forecast_curve                   # 当前窗一次 forward 的 H 步（DTW 用）

run_detect:
  compute_anomaly_assessment ← step1 残差 / detection_curve 统计
  link_anomaly_to_typhoon    ← forecast_curve → DTW；KB 时空初筛不变
```

| 字段 | 用途 | 消费者 |
|------|------|--------|
| `detection_curve` / `current_curve` | 过去–现在异常形态 | 3σ、`anomaly_segments`、曲线色带 |
| `forecast_curve` | 未来 24h 态势形态 | DTW 重排、Top-K 卡片、LLM 报告 |
| `pred_future` | 原始 H 步风/浪 | 前端 forecast 子图、评测 |

---

## Phase 清单

| ID | Goal | 内容 | 验收 |
|----|------|------|------|
| S1 | A | horizon 24 + npz + model 烟测 | `y.shape`, forward shape |
| S2 | A | train + eval + horizon direct | metrics JSON |
| S3 | B | detect 双曲线 + inference 返回 | 单测 curve_source |
| S4 | B | bridge + API 字段 | `/forecast` JSON |
| S4.5 | B | `test_anomaly_forecast_dtw.py` + eval 占位 | pytest 绿 |
| S5 | A+B | 文档 §Direct24h + submission + 答辩脚本 | 无 legacy 矛盾表述 |

---

## 建议 `/subgoal`

```text
/subgoal link_anomaly_to_typhoon 增加参数 dtw_curve: forecast|detection，默认 forecast

/subgoal API 保留 current_curve 作为 detection_curve 别名，deprecated 文档说明

/subgoal 新增 docs 图：双曲线数据流 mermaid，供论文 §5 粘贴

/subgoal forecast_dtw_eval 先 smoke：1 个 DEMO 事件 + mock pred_future，JSON 占位 forecast_topk_hit
```

---

## 文档必须更新的文件

| 文件 | 改动 |
|------|------|
| `docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md` | 新增 **§4 Direct24h 双曲线 DTW**；层 A/B/C 三分；旧 residual→DTW 标 legacy |
| `docs/工程手册/离线系统_预处理数据归档.md` | §4.3 两曲线 + forecast DTW |
| `docs/开发规划/答辩演示脚本.md` | 口播：「24h 预测曲线匹配历史台风」 |
| `submission/tables/anomaly_metrics_direct24h.md` | MAE 表 + DTW 曲线来源脚注 |
| `submission/tables/anomaly_forecast_dtw_eval.json` | 占位指标（可选） |

---

## 答辩口径（完成后）

> 系统做 **24h 直接多步预测** 服务**提前预警**；**在线异常分级**仍看 **下一时刻（3h）预测残差**，保证与观测对齐、可解释。  
> **历史台风类比（DTW）** 改用 **未来 24h 预测曲线** 与 KB 中历史事件曲线匹配，回答「若按当前态势发展，最像哪场过去台风」——与预警定位一致。  
> Oracle 时空 Recall 仍为检索链验收，**不等于**端到端台风识别率；MAE 与 DTW 分表汇报。

---

## 关键代码索引

| 用途 | 路径 |
|------|------|
| DTW 重排 | `src/anomaly/detect.py` — `rerank_candidates_by_dtw`, `link_anomaly_to_typhoon` |
| 曲线组装 | `src/anomaly/eddy_typhoon_bridge.py` — `_current_curve_for_detect` → 拆双曲线 |
| 推理 | `src/anomaly/inference.py` |
| NC 在线 | `src/anomaly/windwave_nc_bridge.py` |
| API | `web_api/routers/windwave_report.py` |
| 台风 eval | `scripts/anomaly_typhoon_link_eval.py` |
| 旧 Goal | 本文档替代 `GOAL_风浪Direct多步预测.md` 中单曲线 S4 表述 |
