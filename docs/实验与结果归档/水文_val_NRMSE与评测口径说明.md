# 水文 val NRMSE 现象、评测口径与预处理说明

> **归档日期**：2026-05-15  
> **数据依据**：`AutoDL/outputs/hydro_l0_eos003/hydro_compare_val_summary_eos003.json`（L2 基线）、`src/hydro/eval.py`、`docs/架构与方法/命题方数据集说明.md`  
> **用途**：答辩/论文中解释「温盐 NRMSE 小、u/v 大」；指导物理空间 NRMSE 与命题方对齐；避免把 z-space 四通道平均误判为赛题结论。

---

## 1. 归档：val 上 temp/sal NRMSE 小、u/v 大的机制

### 1.1 指标定义（仓库实现）

`src/hydro/eval.py` / `hydro_cloud_assessment.py compare` 在 **已 z-score 的目标 y** 上：

```text
RMSE_c  = sqrt( mean( (pred - y)^2 ) )   # 对 B×T×H×W 聚合
NRMSE_c = RMSE_c / mean( |y_c| )
nrmse_avg = 四通道 NRMSE 算术平均
```

**不是**物理量纲上的「相对真值」，也 **不是**命题方口头描述的「先区域平均、再全时次平均」的唯一定义（需官方脚本确认）。

### 1.2 同一次 val（L2）上的数量级（说明「反差」）

| 通道 | NRMSE | RMSE (z) | mean(\|y\|) z-space | 物理 RMSE（×std） |
|------|-------|----------|---------------------|-------------------|
| temp | **0.070** | 0.062 | **0.888** | ~0.59 |
| sal  | **0.033** | 0.033 | **1.001** | ~0.53 |
| u    | **1.647** | 0.093 | **0.057** | ~0.057 m/s |
| v    | **1.551** | 0.806 | **0.520** | ~0.059 m/s |

- **RMSE(z)**：u 与 temp **同量级**（~0.06～0.09），并非差一个数量级。  
- **mean(\|y\|)**：u 的分母约为 temp 的 **1/16** → NRMSE 被放大。  
- **物理 RMSE**：四通道都在 ~0.05～0.6 的合理量级，**不存在「u 物理误差比 temp 差 20 倍」**。

**结论（观感）**：val 上 **u/v NRMSE 显得很大**，首要来自 **z-score 后分母 mean(\|y\|) 过小（尤其 u）**，叠加 **v 的 RMSE(z) 本身偏大**；不是单通道物理预报突然崩溃。

### 1.3 归档：分母 mean(\|y\|) 偏小的可能原因

| 类别 | 机制 | 对 u/v 的影响 |
|------|------|----------------|
| **分布形状** | z-score 后仍大量格点 \|y\|≈0（弱流、正负抵消） | **u 最明显**（val mean\|y\|≈0.057） |
| **归一化** | 训练集全局 σ_u 偏大 → 主体弱流被压到 \|y\|≪1 | 主要拉低 **u** |
| **划分漂移** | μ、σ 仅在 train(1994–2013) 拟合，val=2015 更静/更弱 | val 上 \|y\| 可系统性偏小 |
| **缺测处理** | `nan_to_num→0` 增加 \|y\|=0 格点 | 流场缺测多时可放大 |
| **预报窗** | 72 h 输出；流场时变剧烈处 RMSE 大，但平均 \|y\| 仍可由弱流区主导 | **v** 的 RMSE 与持久性基线都大 |

temp/sal 场更平滑、\|y\| 在 z-space 常接近 0.8～1.0，故 NRMSE 自然落在 0.03～0.07。

### 1.4 归档：预处理 vs 评价指标——谁为主因？

| 问题 | 结论 |
|------|------|
| **数值反差（temp≈0.07 vs u≈1.65）** | **主因**：**z-score 预处理** + **在标准化场上用 mean(\|y\|) 作 NRMSE 分母** 的组合；不是 RMSE 单独恶化 20 倍。 |
| **指标是否「算错」** | **实现自洽、无 bug**；用于 **同实验、同通道** 选 best 可接受。 |
| **指标是否「用对」** | **不宜**：四通道算术平均、与赛题 NRMSE 直接划等号、用 0.15 卡 u 的 NRMSE。 |
| **u/v 是否更难报** | **是**（Skill、Pearson、持久性基线）；应看 **分项 + 物理 RMSE + Skill**，不能只看 z-space NRMSE 绝对值。 |

**一句话**：不是「预处理错了」或「指标公式写错」二选一，而是 **训练用 z-score 合理，但当前 NRMSE 报法不适合跨通道、也未必是赛题口径**。

---

## 2. 怎么做「物理空间 NRMSE / 与命题方对齐」

### 2.1 目标分层

| 层级 | 目的 | 是否必须重训 |
|------|------|----------------|
| **A. 物理空间评估（自评）** | 反标准化后算 RMSE/NRMSE，材料分项陈述 | **否**（同 checkpoint） |
| **B. 与命题方公式对齐** | 区域平均、全时次平均、划分一致 | 以官方脚本为准；可能 **仅改评估** |
| **C. 预处理分支** | 若官方要求在原始量纲或指定归一化上训练 | 可能 **要重训** |

### 2.2 A. 物理空间 NRMSE（仓库内可落地）

**已有能力**

- `data/processed/stats/hydro_zscore.npz`：`mean`、`std`、`features`（与 `temp/sal/u/v` 顺序一致）。  
- `src/hydro/physical_scale.py`：`denorm_array`、反标准化逻辑。  
- `hydro_cloud_assessment.py compare` 的 `raw_*` 中已有 **`rmse_physical_scale` ≈ RMSE(z)×std**（**不是**完整反标准化后的 NRMSE）。

**已实现（2026-05-15）**

- `src/hydro/extended_metrics.py`：提供 `hydro_zscore.npz` 时累计  
  `nrmse_physical_per_feature`、`rmse_physical_per_feature`、`mae_physical_per_feature`、`mean_abs_y_physical_per_feature` 及 `*_physical_avg`。  
- `scripts/hydro_cloud_assessment.py compare`：JSON `summary` 含 `baseline_nrmse_physical_avg` / `experiment_nrmse_physical_avg`；`per_feature` 含 `baseline_nrmse_phys` 等；Markdown 汇总表增加物理 NRMSE 行。  
- 未传 `--stats-npz` 时 **自动尝试** `data/processed/stats/hydro_zscore.npz`。

**命令**

```bash
python scripts/hydro_cloud_assessment.py compare --split val \
  --baseline-config config/hydro_hycom_l2.yaml --baseline-ckpt outputs/hydro_l2/best.pt \
  --experiment-config config/experiments/hydro_hycom_l0_eos003.yaml \
  --experiment-ckpt outputs/hydro_l0_eos003/best.pt \
  --stats-npz data/processed/stats/hydro_zscore.npz \
  --out-summary-json AutoDL/outputs/hydro_l0_eos003/hydro_compare_val_summary_eos003.json
```

### 2.3 B. 与命题方对齐（流程清单）

命题方曾沟通口径（团队记录，**以最新官方文本/脚本为准**）：

- 指标为 **NRMSE**（相对真值）；  
- **先空间（区域）平均，再全时次平均**；  
- **train/val/test 划分须一致**；  
- **总体与分项可同时提供**（避免四通道平均掩盖温盐/流场差异）。

**建议步骤**

| 步骤 | 动作 |
|------|------|
| 1 | 索取或下载 **官方评测脚本 / 样例提交格式**（若有）。 |
| 2 | 用 **同一划分** 的 pred 与 truth（NetCDF 或官方指定数组布局）跑官方脚本，得到 **分项 + 总体** NRMSE。 |
| 3 | 与自建 `eval.py`、compare JSON **逐项 diff**：分母是 mean(\|y\|) 还是别的、平均顺序、是否 z-score。 |
| 4 | 在 `submission/tables/` 增加 **「赛题口径」** 表，与 **「研发口径 z-space」** 表 **分表**，禁止混标题。 |
| 5 | 答辩口径写：**研发训练用 z-score；赛题报数以官方评测为准（或物理空间自评作辅证）**。 |

**若无官方脚本**：用物理空间 NRMSE + 分项表作为自评；向命题方发 **1 页公式对照**（附本仓库 `eval.py` 公式）请求确认。

### 2.4 C. 预处理若需与赛题一致

见下文 **§3**；仅在官方明确要求 **禁止 z-score** 或指定归一化时，开 **新 config 分支** 重跑预处理与训练，**勿覆盖** 当前可复现 `hydro_zscore.npz` 基线。

---

## 3. 命题是否要求预处理？本仓库怎么做？

### 3.1 命题方书面要求（摘要）

依据 `docs/架构与方法/命题方数据集说明.md`（原文 `服创数据集/数据集说明.md`）：

| 有明确要求 | 无明确要求 |
|------------|------------|
| 海区、分辨率、变量 **SST/SSS/SSU/SSV**（不得私自加变量） | **未规定** 必须 z-score、min-max 或物理单位训练 |
| 输入 **n 小时 + 当前时刻**，输出 **未来 72 小时** | **未规定** 滑窗 stride、缺测填 0 等细节 |
| 划分：train 1994–2013 / test 2014 / val 2015 | **未给出** 官方 NRMSE 计算代码 |
| 指标名称：**NRMSE**（归一化 RMSE） | 归一化分母、空间/时间平均顺序需 **沟通或脚本对齐** |

**结论**：命题要求的是 **数据范围、变量、窗长、划分与评测指标名称**；**未强制** 本仓库的 z-score 预处理。预处理属于 **参赛方工程实现**，但 **报数必须与评测口径一致**。

### 3.2 本仓库水文训练预处理（标准链路）

详见 [离线系统_预处理数据归档.md](../工程手册/离线系统_预处理数据归档.md) §5.1。

| 步骤 | 操作 |
|------|------|
| 1 | `raw_root` 扫描 `海域要素预测/**/*.nc`，`variable_map.yaml` 映射 → `temp/sal/u/v` |
| 2 | 按年 `hydro_year_split` 划分 train/val/test |
| 3 | time 拼接 → `(T,H,W,4)`，缺测 **nan_to_num→0** |
| 4 | `build_windows`：`input_steps=168`，`output_steps=72` → `X/y` 滑窗 npz |
| 5 | **仅在训练滑窗上** `zscore_fit` → `data/processed/stats/hydro_zscore.npz` |
| 6 | train/val/test **共用** 该 mean/std → `data/processed/hydro/X_*.npz` |

**命令**：

```bash
python -m src.preprocess.hydro_dataset --config config/hydro_hycom.yaml \
  --from-nc --data-config config/data.yaml --year-split
```

在线上传 NC 演示：优先用已保存的 `hydro_zscore.npz`；缺失时 **按本次上传现算** stats（与离线训练不一致时 NRMSE 仅作演示参考）。

---

## 4. 仅对 u/v 做「鲁棒缩放」能否大幅缓解 val NRMSE？

### 4.1 短答

**不能指望「只改 u/v 鲁棒缩放、不重训」就大幅压低 val NRMSE**；  
**在重训 + 只改 u/v 归一化** 的前提下，可能 **明显降低 u 通道 NRMSE 的「分母效应」**，但：

- **四通道平均 NRMSE** 仍多由 u/v 主导，降幅有限；  
- **赛题若用物理空间另一套公式**，改鲁棒缩放不等于自动对齐；  
- **预报难度**（Skill、v 的大 RMSE）需靠模型与损失，不单靠缩放。

### 4.2 机制说明

| 情况 | 对 NRMSE 的影响 |
|------|-----------------|
| **只改评估、不改训练** | 模型仍在旧 z-space 上优化，**物理 NRMSE 有意义，z-space NRMSE 不变**。 |
| **u/v 改用 median/IQR 等再训练** | σ 不再被极端流场拉大时，z 后 **mean(\|y\|)** 可能上升 → **NRMSE_u 分母变大、比值下降**；RMSE 是否同降取决于拟合。 |
| **temp/sal 仍用原 z-score** | 四通道 **不可直接平均** 比较，需分项报。 |
| **命题方不用 z-score** | 须 **官方口径评估**；鲁棒缩放只是训练技巧，不能替代对齐。 |

### 4.3 若要做实验（建议顺序）

1. **零成本**：compare 已带 **物理 RMSE**；补 **物理 NRMSE** 字段（§2.2）。  
2. **低成本**：`config/experiments/` 复制 yaml，仅 `data.normalize.u/v: robust`（需在 `hydro_nc_stack` 实现并与 `hydro_zscore.npz` 分支命名，如 `hydro_robust_uv.npz`），**小 epoch 烟测** 看 u 的 mean(\|y\|) 与 NRMSE 变化。  
3. **对照**：同一 checkpoint 仅改评估 vs 重训后评估，分开记录，避免混结论。

**经验预期**：u 的 NRMSE 从 ~1.6 **降到 ~1.0 左右** 有可能（分母修正）；**降到 0.15 以下** 需 **赛题口径确认 + 模型/损失/物理误差真实下降**，非单靠鲁棒缩放。

---

## 5. 材料书写建议（可直接引用）

1. **分项报告** temp、sal、u、v 的 NRMSE（研发口径）与物理 RMSE。  
2. **禁止** 仅用四通道平均 NRMSE 得出「整体达标/不达标」。  
3. **说明** 训练采用训练集 z-score；评测与命题方对齐情况见 §2.3。  
4. **流场改进** 用 L0/eos005 的 **u/v Skill、MAE** 与 L2 对比，eos003 不作为主推。

---

## 6. 相关文档

| 文档 | 内容 |
|------|------|
| [AutoDL_outputs_云端结果归档.md](AutoDL_outputs_云端结果归档.md) | 云端 compare 数值表 |
| [水文下一步评估与数据真实性策略.md](../评估与数据策略/水文下一步评估与数据真实性策略.md) | 扩展指标与预处理分支策略 |
| [水文_L2L1L0_实验归档与决策.md](水文_L2L1L0_实验归档与决策.md) | z-score 分母对流场放大、与命题对齐 |
| [命题方数据集说明.md](../架构与方法/命题方数据集说明.md) | 变量、窗长、划分 |
| [离线系统_预处理数据归档.md](../工程手册/离线系统_预处理数据归档.md) | NC→npz→z-score 步骤 |
