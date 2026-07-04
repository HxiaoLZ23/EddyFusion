# AutoDL/outputs 云端结果归档

> **归档日期**：2026-05-15（按本地 `AutoDL/outputs/` 目录快照整理）  
> **云机锚点**：`/root/autodl-tmp/EddyFusion/`（日志与 JSON 内 `baseline_ckpt` 等路径以此为准）  
> **说明**：`AutoDL/` 整树默认 **`.gitignore`**，本文档在 **`docs/`** 可提交，便于交接；指标 JSON/表以本地镜像为准，**权重 `*.pt` 不在此目录**（需云机或单独下载）。  
> **风浪异常**：正式 eval 产物主目录为仓库根 **`outputs/anomaly/`**（与云机 `outputs/anomaly` 同名）；当前本地 **`AutoDL/outputs/` 镜像未必含 `anomaly/`**，以本节 **§6** 与 `outputs/anomaly/metrics_summary_{val,test}.json` 为准。

---

## 1. 目录树与职责

```
AutoDL/outputs/
├── cloud/                          # 专项 A：预处理 audit + L0(默认) vs L2 compare
├── hydro/                          # 主配置 hydro_hycom（L2 结构）eval 指标
├── hydro_l2/                       # 隔离实验 L2 基线 eval
├── hydro_l0_eos005/                # 专项 B：EOS weight=0.05，compare 产物
├── hydro_l0_eos003/                # 专项 B：EOS weight=0.03，compare 产物（含 NRMSE 字段）
├── eddy/                           # 涡旋 3ch YOLO-seg
├── eddy_enh/                       # 涡旋 8ch 增强
├── train/                          # 早期/别名训练 run（args.yaml + results.csv，对照用）
└── anomaly/                        # （可选）云机同步；本地主归档见仓库 outputs/anomaly/
```

**仓库根（与云机同名，非仅 AutoDL 镜像）**

```
outputs/anomaly/                    # 模块 C：风浪异常双头回归 eval + 权重
├── best.pt                         # 新键名 wind_head / wave_head（.gitignore）
├── metrics_summary_val.json        # src.anomaly.eval --split val
├── metrics_summary_test.json       # src.anomaly.eval --split test
└── metrics_summary.json            # 旧版单文件或兼容自检（勿作业务口径）
```

| 子目录 | 配置（仓库内） | 云机 `output_dir`（典型） | 本镜像含权重 |
|--------|----------------|---------------------------|--------------|
| `cloud/` | `hydro_hycom_l0` + `hydro_hycom_l2` | — | 否 |
| `hydro/` | `config/hydro_hycom.yaml` | `outputs/hydro` | 否 |
| `hydro_l2/` | `config/hydro_hycom_l2.yaml` | `outputs/hydro_l2` | 否 |
| `hydro_l0_eos005/` | `config/experiments/hydro_hycom_l0_eos005.yaml` | `outputs/hydro_l0_eos005` | 否 |
| `hydro_l0_eos003/` | `config/experiments/hydro_hycom_l0_eos003.yaml` | `outputs/hydro_l0_eos003` | 否 |
| `eddy/` | 3ch 数据集 yaml | `AutoDL/outputs/eddy` 或 `outputs/eddy` | 否 |
| `eddy_enh/` | `config/eddy_enh.yaml` 等 | `AutoDL/outputs/eddy_enh` | 否 |
| `anomaly/`（云镜像，可选） | `config/anomaly.yaml` | `outputs/anomaly` | 否 |
| **`outputs/anomaly/`**（本地主路径） | 同上 | 仓库根 `outputs/anomaly` | 否 |

**拷回建议**（训练结束后）：`best.pt`、`metrics_summary*.json`、`figures/`、compare 的 `*_summary*.json` 与 `*_vs_l2_*.md`；命令见 [水文_云端归档与专项B启动.md](../工程手册/水文_云端归档与专项B启动.md)、[云端训练与目录归档.md](../工程手册/云端训练与目录归档.md) §5。  
**风浪异常**：`outputs/anomaly/best.pt` + `metrics_summary_val.json` / `metrics_summary_test.json`；汇总表 `python scripts/anomaly_export_metrics_table.py` → `submission/tables/anomaly_metrics_val_test.{md,csv}`。

---

## 2. 专项 A：`cloud/`（数据可信 + L0 默认 vs L2）

| 文件 | 含义 |
|------|------|
| `hydro_preprocess_audit.json` | `hydro_cloud_assessment.py audit`；**`issues: []`** → 预处理/形状/统计量自检通过 |
| `hydro_compare_val_summary.json` | **实验** `outputs/hydro_l0/best.pt` vs **基线** `outputs/hydro_l2/best.pt`，split=val |
| `hydro_compare_test_summary.json` | 同上，split=test |
| `hydro_l0_vs_baseline_val.{md,csv}` | 材料表（val） |
| `hydro_l0_vs_baseline_test.{md,csv}` | 材料表（test） |

### 2.1 L0（默认 EOS=0.1）相对 L2 — 汇总

| split | MAE_avg（实验） | Skill_avg（实验） | Pearson_avg（实验） | `material_line` |
|-------|-----------------|-------------------|---------------------|-----------------|
| val | 0.1289（↓） | -2.168（**劣于** L2 -2.046） | 0.746（↑） | **false** |
| test | 0.1354（↓） | -2.022（**劣于** L2 -1.961） | 0.740（↑） | **false** |

分项规律（与专项 B 一致）：**u/v** 误差与 Skill 多优于 L2；**temp/sal** 在 val 上易拖低 Skill（持久性基线极强）。

---

## 3. 水文基线：`hydro_l2/`、`hydro/`

### 3.1 L2 独立 eval（`src.hydro.eval` 口径）

| split | `nrmse_avg` | `passed`（四通道均值 NRMSE≤0.15） |
|-------|-------------|-------------------------------------|
| val | **0.8252** | false |
| test | （见 `metrics_summary_test.json`） | false |

| 通道 | val NRMSE |
|------|-----------|
| temp | 0.0698 |
| sal | 0.0333 |
| u | 1.6467 |
| v | 1.5509 |

> 赛题阈值 0.15 针对**单通道**；u/v 在 z-score 空间上 `mean(|y|)` 很小，NRMSE 数值常远大于 0.15，**不宜仅用四通道算术平均判定整体达标**（见命题方口径与 [水文下一步评估与数据真实性策略.md](../评估与数据策略/水文下一步评估与数据真实性策略.md)）。

`hydro/` 下 `metrics_summary_*.json` 与 L2 数值一致（同为 level 2 结构主配置产物），作对照备份。

---

## 4. 专项 B：EOS 权重对照（实验 vs L2）

对比脚本：`python scripts/hydro_cloud_assessment.py compare`（扩展指标；003 批次 JSON **含 `baseline_nrmse_avg` / 分项 NRMSE**）。

### 4.1 总体（四通道平均）

| 实验 | EOS weight | split | MAE_avg | NRMSE_avg | Skill_avg | Pearson_avg | material_line |
|------|------------|-------|---------|-----------|-----------|-------------|-----------------|
| **eos005** | 0.05 | val | 0.1290 ↓ | —（JSON 未写入 summary，可重跑 compare） | -2.122 | 0.744 ↑ | false |
| **eos005** | 0.05 | test | 0.1347 ↓ | — | **-1.943 ↑** | 0.741 ↑ | **true** |
| **eos003** | 0.03 | val | 0.1294 ↓ | **0.8157 ↓** | -2.330 | 0.741 ↑ | false |
| **eos003** | 0.03 | test | 0.1357 ↓ | **0.8315 ↓** | -2.129 | 0.736 ↑ | false |
| L2 基线 | — | val | 0.1301 | 0.8252 | -2.046 | 0.730 | — |
| L2 基线 | — | test | 0.1365 | 0.8413 | -1.961 | 0.727 | — |

**选型结论（当前镜像）**

- **相对 L2**：005、003 均在 **MAE / NRMSE（003）/ Pearson** 上略优，**Skill_avg 仅 eos005 在 test 上优于 L2**。
- **005 vs 003**：**005 全面更优**（test 上唯一满足 `material_line_l0_stable_improve`）；003 的 **sal** 分项 Skill 劣于 L2 更明显。
- **推荐对外主 checkpoint（L0 调参线）**：`outputs/hydro_l0_eos005/best.pt`（权重在云机，需单独下载）。

### 4.2 分项要点（eos003 vs L2，test）

| 通道 | MAE | NRMSE | Skill | 相对 L2 |
|------|-----|-------|-------|---------|
| temp | 略优 | 略优 | 略优 | 小幅改善 |
| sal | 差 | 差 | 更负 | **拖后腿** |
| u | 优 | 优 | 优 | 明确改善 |
| v | 优 | 优 | 优 | 明确改善 |

明细表：`hydro_l0_eos003/hydro_l0_eos003_vs_l2_{val,test}.md`；JSON：`hydro_compare_*_summary_eos003.json`。

### 4.3 训练过程（eos005 / eos003）

- **eos005**（云机日志摘要）：计划 60 epoch，**约 epoch 38** 验证 NRMSE 最优（≈0.589），**epoch 48 早停**（patience=10）；`train_loss` 仍下降而 val 已平台。
- **eos003**：本地镜像**无** `metrics_summary.json` / 训练曲线；若有终端截图可存为 `hydro_l0_eos003/train_log.png` 并在此节补一行 best_epoch。

### 4.4 各实验目录文件清单

| 路径 | 已有文件 |
|------|----------|
| `hydro_l0_eos005/` | `hydro_compare_val_summary_eos005.json`、`hydro_compare_test_summary_eos005.json` |
| `hydro_l0_eos003/` | 上表 + `hydro_l0_eos003_vs_l2_val.md`、`hydro_l0_eos003_vs_l2_test.md` |

---

## 5. 涡旋：`eddy/`、`eddy_enh/`

|  run | split | mask_map50 | `passed` |
|------|-------|------------|----------|
| eddy（3ch） | val | 0.762 | — |
| eddy（3ch） | test | 0.730 | — |
| eddy_enh（8ch） | val | **0.839** | true |
| eddy_enh（8ch） | test | **0.826** | true |

另含 `train/args.yaml`、`train/results.csv`（Ultralytics 训练曲线数据源）。口径说明见各 JSON 内 `note`（与命题方 IoU 需人工核对）。

---

## 6. 风浪异常：`outputs/anomaly/`（模块 C）

**模块定位**：**风浪异常识别为中心**（LSTM 一步预测 + 残差/3σ 分级）；**台风为弱关联解释层**（KB 检索 + 可选 DTW）。分层指标见 [风浪异常_指标口径与台风关联评测.md](风浪异常_指标口径与台风关联评测.md)。

| 层级 | 指标 | 入口 |
|------|------|------|
| A. 预测底座 | MAE / RMSE（val+test） | `python -m src.anomaly.eval` |
| A′. 补充超前 | 24h rollout MAE（区域平均） | `python scripts/anomaly_eval_horizon24.py` |
| B. 台风关联 | 关联 Recall（测试年，Oracle 查询框） | `python scripts/anomaly_typhoon_link_eval.py` |

评估入口：`python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split val|test`。  
**业务口径**须为 **新键名** `wind_head` / `wave_head` 的 `best.pt`；`passed` 为占位规则 **`mae_avg < 0.5`**，**不得**当作赛题「真实台风中被正确识别的比例」（见 `src/anomaly/eval.py` 模块说明）。

### 6.1 val / test 回归指标（`metrics_summary_{val,test}.json`）

下表为 **`outputs/anomaly/metrics_summary_val.json` / `metrics_summary_test.json`** 归档数值（与 `submission/tables/anomaly_metrics_val_test.*` 一致；**材料对外以本表为准**）。快照：**2026-05-20** 在本地仓库根对当前 `outputs/anomaly/best.pt` 重跑 `eval` 后导出。

| split | MAE 风速 | MAE 波高 | MAE 平均 | RMSE 风速 | RMSE 波高 | RMSE 平均 | `passed` |
|-------|----------|----------|----------|-----------|-----------|-----------|----------|
| val | 0.0501 | 0.0089 | **0.0295** | 0.0644 | 0.0113 | 0.0378 | true |
| test | 0.0495 | 0.0088 | **0.0292** | 0.0628 | 0.0110 | 0.0369 | true |

> **历史快照（2026-05-18，云机/旧 NPZ 口径，已 superseded）**：val `mae_avg`≈0.27、test≈0.46。若与上表差异大，以**本机当前 NPZ + 双头 `best.pt` + split JSON** 为准；勿用无 split 的 `metrics_summary.json`（旧 `head.*` 自检可出现 MAE 风 ~4+）。

**解读要点**

- val / test 的 **`mae_avg` 均 < 0.5**，与 eval 占位 `passed=true` 一致。
- test 上 MAE 略高于 val，属常见划分差异；答辩材料应 **同时报 val 与 test**，勿只摘一侧。
- 若 val 上出现 **MAE 风速 ~4+、MAE 平均 ~5+**，多为 **`metrics_summary.json`（无 split 后缀）或旧格式 `head.*` 权重** 的兼容自检，**不得**当作业务指标（见 [风浪异常模块_交付证据与复现.md](风浪异常模块_交付证据与复现.md)、[后续开发工作清单_未完成项与云端L0专项.md](../开发规划/后续开发工作清单_未完成项与云端L0专项.md) §1.3.1）。
- **持续性基线**（val，最后一格预测下一时刻）：平均 MAE ≈ **0.063**；当前模型 ≈ **0.030**，说明任务为区域平均短超前一步，数值小不等于「台风识别率 97%」。
- **24h 补充评估**见 **§6.5**（自回归 rollout，MAE 量级 ~0.16，更接近长超前物理直觉）。

### 6.5 24h 超前回归（区域平均，自回归 rollout）

**定位**：**补充口径**，与 §6.1 同权重 `best.pt`（训练目标仍为 **3h 一步**）；24h 通过 **8 步自回归 rollout** 评估，**非** 24h 直接监督重训。

快照：**2026-06-05** 本地 `python scripts/anomaly_eval_horizon24.py --split both --horizon-hours 24`；JSON：`outputs/anomaly/metrics_summary_{val,test}_horizon24h.json`；汇总 `outputs/anomaly/eval_horizon24h_summary.json`；材料表 `submission/tables/anomaly_metrics_horizon24h.md`。

| split | MAE 风速 | MAE 波高 | MAE 平均 | RMSE 平均 | 持续性 MAE 平均 | MAE/持续性 | `n_samples` |
|-------|----------|----------|----------|-----------|-----------------|------------|-------------|
| val | 0.2429 | 0.0706 | **0.1567** | 0.2005 | 0.1743 | 0.899 | 2897 |
| test | 0.2480 | 0.0730 | **0.1605** | 0.2075 | 0.1854 | 0.866 | 2905 |

**解读要点**

- 量纲：**\|U10\| m/s、SWH m**，预处理与 eval **均无 StandardScaler**（与 §6.1 审计结论一致）。
- 相对 §6.1（3h 一步）：MAE 平均由 **~0.030** 升至 **~0.16**，长超前难度显著增加。
- 相对持续性（窗口末值外推 24h）：模型 **略优**（比值 **0.87～0.90**），提升有限——当前权重 **未按 24h 监督训练**。
- 答辩：**主表仍用 §6.1（3h 训练口径）**；§6.5 用于回应「MAE 过小是否未反归一化」——长超前下误差回到 **0.15～0.25 m/s（风）** 量级。
- 若需 **24h 直接监督**主指标：须 `config/anomaly.yaml` → `horizon_hours: 24` 重建 NPZ 并重训（当前仓库 **未做**）。

复现：

```bash
python scripts/anomaly_eval_horizon24.py --split both --horizon-hours 24
```

### 6.2 目录文件与材料链

| 路径 | 含义 |
|------|------|
| `outputs/anomaly/best.pt` | 对外主 checkpoint（新双头） |
| `outputs/anomaly/metrics_summary_val.json` | val eval 原始 JSON |
| `outputs/anomaly/metrics_summary_test.json` | test eval 原始 JSON |
| `outputs/anomaly/metrics_summary_{val,test}_horizon24h.json` | **24h rollout** 补充 eval（§6.5） |
| `outputs/anomaly/eval_horizon24h_summary.json` | 24h val+test 汇总 |
| `submission/tables/anomaly_metrics_horizon24h.md` | 24h 指标表（与 §6.5 同步） |
| `outputs/anomaly/metrics_summary.json` | 历史单文件；**val MAE≈4.26、mae_avg≈3.77、`passed=false`** 时为旧口径，仅作兼容记录 |
| `submission/tables/anomaly_metrics_val_test.md` | 由 `scripts/anomaly_export_metrics_table.py` 从 val/test JSON 生成（与 §6.1 同步） |
| `submission/tables/anomaly_metrics_val_test.csv` | 同上 |
| `docs/实验与结果归档/风浪异常_典型案例对照.md` | 答辩案例模板（时空窗/台风 ID 赛前补全） |
| `docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md` | 层 A/B 指标定义与答辩表述 |
| `submission/tables/anomaly_typhoon_link_recall_2024.*` | 层 B 关联 Recall（默认 test 年 2024） |

### 6.3 复现 eval 与导出表（仓库根）

```bash
python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split val
python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split test
python scripts/anomaly_export_metrics_table.py
python scripts/anomaly_eval_horizon24.py --split both --horizon-hours 24
```

云机训练结束后，将上述 JSON（及可选 `best.pt`）拷至本地 **`outputs/anomaly/`**；若统一放入 `AutoDL/outputs/anomaly/`，可在 §1 树中补一行镜像路径，数值仍以 JSON 为准。

### 6.4 台风关联 Recall（层 B，不重训）

在 **真值台风时空窗 = 查询框（Oracle）** 下评测 `link_anomaly_to_typhoon` Top-K 是否命中 `event_id`。**不是**端到端格点 3σ → 自动报台风 Recall。答辩：**MAE 主讲（§6.1）**，本小节**弱讲**。

**IBTrACS 验收快照（2026-05-21，本机已跑通）**

| 项目 | 数值 |
|------|------|
| KB 构建 | `build_typhoon_kb.py` → `events_count` **13544** |
| 测试年 / 过滤 | **2024**，峰值风速 ≥ **34 kt** |
| Top-K | **10** |
| 命中 / 真值 | **84 / 84** |
| **关联 Recall** | **1.000** |

与论文 **表 7-3**、`submission/tables/anomaly_typhoon_link_recall_2024.md` 一致。因 Oracle 查询框即真值框，Recall=1.0 表示**检索链路与索引验收通过**，**不得**在摘要中写成「台风识别准确率 100%」。端到端 POD **未做**。

```bash
python scripts/build_typhoon_kb.py --source-csv data/raw/typhoon/ibtracs/ibtracs.ALL.list.v04r01.csv
python scripts/anomaly_typhoon_link_eval.py --test-years 2024 --top-k 10 --min-peak-wind-kt 34
```

典型案例（可解释性，见 [风浪异常_典型案例对照.md](风浪异常_典型案例对照.md)）：**ANGGREK**（2024012S09093）、**JASPER**（2023337S08165）；图示 `outputs/anomaly/figures/202401_continuous.png`、`year2024_continuous.png`（论文图 5-1 / 5-2）。

---

## 7. 缺失与待补（云机 → 本地镜像）

| 项 | 说明 |
|----|------|
| 各实验 `best.pt` / `last.pt` | `.gitignore` 忽略；答辩演示需从云机 `outputs/hydro_l0_eos00*/` 下载 |
| `hydro_l0_eos005` 的 NRMSE 汇总字段 | 旧版 compare JSON；可用当前脚本 **重跑 compare** 与 003 字段对齐 |
| `figures/`、训练 log | 未同步；可选拷回 `AutoDL/outputs/<exp>/figures/` |
| `outputs/cloud/` 与 `hydro_l0_eos00*/` 副本 | 可将 `*_eos003.json` 再拷一份到 `cloud/` 统一命名（可选） |
| `outputs/anomaly/best.pt` | 与 JSON 同目录；演示/答辩需单独下载或本地训练产出 |
| `submission/tables/anomaly_metrics_val_test.*` 与 JSON 不一致 | 对当前 `best.pt` 重跑 §6.3 三条命令 |
| `AutoDL/outputs/anomaly/` | 当前镜像可能为空；可选从云机同步，与仓库 `outputs/anomaly/` 二选一归档即可 |

---

## 8. 复现 compare（bash，云机仓库根）

```bash
# eos003（示例）
python scripts/hydro_cloud_assessment.py compare --split val \
  --baseline-config config/hydro_hycom_l2.yaml --baseline-ckpt outputs/hydro_l2/best.pt \
  --experiment-config config/experiments/hydro_hycom_l0_eos003.yaml \
  --experiment-ckpt outputs/hydro_l0_eos003/best.pt \
  --stats-npz data/processed/stats/hydro_zscore.npz \
  --out-table-md submission/tables/hydro_l0_eos003_vs_l2_val.md \
  --out-summary-json AutoDL/outputs/hydro_l0_eos003/hydro_compare_val_summary_eos003.json
```

续行用 **反斜杠 `\`**，勿用 PowerShell 反引号。

---

## 9. val NRMSE 解读（温盐小、u/v 大）

完整归档（机制表、分母原因、预处理 vs 指标主因、物理 NRMSE 与命题对齐、鲁棒缩放预期）见：

**[水文_val_NRMSE与评测口径说明.md](水文_val_NRMSE与评测口径说明.md)**

摘要：当前 compare/eval 的 NRMSE 在 **z-score 场** 上计算，**u 的 mean(\|y\|)≈0.057** 导致 NRMSE 放大；**物理 RMSE** 四通道同量级。材料应 **分项报**，勿仅用四通道平均；赛题 NRMSE 须 **官方脚本或公式确认**。**多指标与「能用」结论**见 [水文_其他指标与能用标准归档.md](水文_其他指标与能用标准归档.md)。

---

## 10. 相关文档

| 文档 | 内容 |
|------|------|
| [水文_val_NRMSE与评测口径说明.md](水文_val_NRMSE与评测口径说明.md) | **§1–§4** val NRMSE 三段归档 + 物理评测与预处理问答 |
| [水文_云端归档与专项B启动.md](../工程手册/水文_云端归档与专项B启动.md) | audit/compare 命令与 eos005/003 模板 |
| [后续开发工作清单_未完成项与云端L0专项.md](../开发规划/后续开发工作清单_未完成项与云端L0专项.md) | 专项 A/B 勾选与 §3.0 摘要 |
| [水文_L2L1L0_实验归档与决策.md](水文_L2L1L0_实验归档与决策.md) | L2/L1/L0 隔离训练决策 |
| [云端训练与目录归档.md](../工程手册/云端训练与目录归档.md) | 路径契约与训练入口 |
| [风浪异常模块_交付证据与复现.md](风浪异常模块_交付证据与复现.md) | 模块 C eval 命令、JSON 字段、材料表路径 |
| [风浪异常_典型案例对照.md](风浪异常_典型案例对照.md) | 答辩案例结构模板 |
| `outputs/cloud/README.md` | 仓库内归档文件名约定（可提交） |

**维护**：每次云机大批量下载后，更新本文 **§4 表格**、**§6.1 风浪指标** 与 **§7 缺失项**，并在 `开发阶段文档/训练与实验记录.md`（若使用）记一行实验 ID + 结论。NRMSE 口径变更时同步 **[水文_val_NRMSE与评测口径说明.md](水文_val_NRMSE与评测口径说明.md)**。风浪模块 JSON 更新后执行 **§6.3** 刷新 `submission/tables/anomaly_metrics_val_test.*`。
