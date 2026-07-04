# V6 涡旋解耦实验 Phase A 烟测归档

> 规范原文：`开发阶段文档/基于时空一致性弱监督标签的中尺度涡旋实例分割训练方案v6.md`  
> 独立实验线，**不替换** `eddy_cloud_fair` 主结论路径。  
> 归档日期：2026-06-06

---

## 1. Phase A 锁定配置

| 项 | 值 |
| --- | --- |
| Teacher / 标签 | OW(P24)，`--single-percentile 24`，`time_stride=1` |
| train NC | `20130101_20221231.nc` |
| train 日历 | **2018-01-01 ~ 2018-03-31**（90 帧） |
| val NC | `20240101_20241231.nc`（去首尾约 **364** 帧） |
| ADT 归一化 | **方案 A**：每时间片、每变量独立 P2/P98 → clip → uint8 |
| Fair 输入 | `[norm(ADT(t)), norm(ADT(t)), norm(ADT(t))]` |
| Proposed 输入 | `[norm(ADT(t-1)), norm(ADT(t)), norm(ADT(t+1))]`（±1 日） |
| Leakage 输入 | ADT/U/V 各通道独立方案 A 伪彩（旁证） |
| 训练 | YOLOv8n-seg，640，batch=4，**epoch=50**，seed=42 |
| 评测 | 仅 **val 2024**（禁止 test 2023） |

**train/val 规模对比**：train 90 帧 vs val 364 帧——训练量远小于验证量，结果仅宜作**趋势判断**。

---

## 2. 数据与产物

| 目录 | train | val | labels |
| --- | ---: | ---: | --- |
| `AutoDL/dataset/eddy_v6_leakage` | 90 | 364 | 主 labels 树 |
| `AutoDL/dataset/eddy_v6_fair` | 90 | 364 | 自 leakage 复制 |
| `AutoDL/dataset/eddy_v6_proposed` | 90 | 364 | 自 leakage 复制 |

| 组别 | config | 权重 |
| --- | --- | --- |
| Fair | `config/eddy_v6_fair.yaml` | `outputs/eddy_v6_fair/last.pt` |
| Proposed | `config/eddy_v6_proposed.yaml` | `outputs/eddy_v6_proposed/last.pt` |
| Leakage | `config/eddy_v6_leakage.yaml` | `outputs/eddy_v6_leakage/last.pt` |

导出：`scripts/run_eddy_v6_phase_a_export.ps1`  
主表：`submission/tables/eddy_v6_fair_vs_proposed.md`  
Rule 旁证：`submission/tables/eddy_v6_rule_compare.md`  
生命周期：`submission/tables/eddy_v6_lifetime_val.md`

---

## 3. val 主表（2024，50 epoch）

| 组别 | mAP50 | mAP50-95 | P | R | F1 | 参与排名 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Fair** | **0.705** | 0.323 | **0.690** | 0.619 | 0.652 | ✓ 主比 |
| **Proposed** | 0.680 | 0.304 | 0.628 | **0.646** | 0.636 | ✓ 主比 |
| Leakage | 0.641 | 0.264 | 0.617 | 0.617 | 0.614 | ✗ 旁证 |
| Rule (OW P24) | 1.000† | — | 1.000 | 1.000 | 1.000 | ✗ Teacher |

† Rule 为相对伪标签 instance IoU≥0.5 重放的 matched mIoU，**非 YOLO mAP**。

**Proposed − Fair**：ΔmAP50 = **−0.025**；ΔP = **−0.062**；ΔR = **+0.027**

### 3.1 分类别 mAP50（val，Ultralytics eval 日志）

| 组别 | 气旋 (CE) | 反气旋 (AE) |
| --- | ---: | ---: |
| Fair | 0.733 | 0.677 |
| Proposed | 0.746 | 0.614 |
| Leakage | 0.731 | 0.552 |

- 气旋：三组接近（Fair / Leakage ≈ 0.73，Proposed 略高 0.746）。
- 反气旋：Fair 最高；Leakage 明显偏低（0.552），是 Leakage 总 mAP 低于 Fair 的主要来源之一。

---

## 4. 结果解读（Phase A 口径）

### 4.1 Leakage < Fair：与 `eddy_cloud_fair` 不可直接对比

Phase A 观察到 **Leakage（0.641）< Fair（0.705）**，即在 **P24 Teacher + 方案 A 导出 + 90 帧短训** 下，含 U/V 的伪彩输入未高于纯 ADT×3。

**不得**由此推出「U/V 对涡旋识别无效」。原因：

1. **Rule（P/R≈1）才是 OW Teacher 上界**；Leakage 仍是 YOLO 学 RGB→mask，不是规则重放。
2. 与 **`eddy_cloud_fair` val mAP≈0.93** 差异来自：train 规模（90 帧 vs 多年）、标签/导出（P24 vs 原 cloud 配置）、归一化（方案 A vs 原伪彩）——**非同一实验**。
3. 短训下 **ADT 单帧已携带与 OW 伪标签相关的几何信息**；U/V 伪彩编码未必比「ADT×3」更易学。
4. 反气旋上 Leakage 偏弱，可能反映三通道独立分位后 U/V 语义与类别判别未对齐，而非 U/V 无物理意义。

**建议论文表述**（归档定稿）：

> Leakage 组在 Phase A 中未高于 Fair 组，并不说明 UGOS/VGOS 对涡旋识别无效。该实验仅在 2018Q1 的 90 帧小样本、P24 Teacher 和统一 50 epoch 配置下进行，结果主要反映短训条件下不同输入编码的可学习性。Rule 重放才是 OW Teacher 的理论上限，Leakage 组只是同源输入参考而非上界模型。该结果表明，在小样本烟测条件下，单帧 ADT 输入已经能够提供较稳定的涡旋几何信息，而含 U/V 的伪彩输入并未立即转化为更高的 YOLO 拟合能力。

### 4.2 Fair vs Proposed：初步负结果（主比较）

- 三时刻 ADT（±1 日）**未优于** 单帧 ADT×3：mAP50 −0.025，P −0.062，R +0.027。
- 行为模式：**更激进**——多检出（R↑）伴随误检增加（P↓），典型 precision–recall 权衡，而非全面增益。
- 气旋 mAP Proposed 略高，反气旋 mAP Proposed 低于 Fair；总均值上 Fair 仍优。

**可写、但需加限定语的主句**：

> 在当前训练规模（2018Q1，90 帧）与 OW(P24) Teacher 定义下，连续三时刻 ADT（±1 日）输入**未表现出**优于单时刻 ADT 输入的整体优势；Proposed 召回略升、精度与 mAP 略降。这是**有效负结果（negative result）**，尚不足以作为论文终局结论。

**不宜在 Phase A 直接写死**：「ADT(t) 已包含绝大部分 OW 信息」——更稳妥为「单帧 ADT 在短训下已能部分拟合 OW Teacher，额外 ±1 日通道尚未带来净增益」。

### 4.3 Phase A 已证明 vs 尚未证明

| 已证明 | 尚未证明 |
| --- | --- |
| 导出 / 训练 / eval / 四层评价体系通路跑通 | 三时刻 ADT 在充分 train 下仍不优于 Fair |
| 同 stem 共用 P24 labels 可复现 | ±1 日 triplet 相似度过高是否为根因（待 ±3/±5 消融） |
| 短训下 Fair > Proposed、Fair > Leakage 的趋势 | Leakage 随数据量增大能否追上 Fair |
| Rule Teacher 上界 P/R≈1 | test 2023 终评（Phase B 一次） |

### 4.4 对 V6 研究叙事的调整（Phase A 后）

- **原核心假设**：去除 U/V 后，连续 ADT 时序可补偿信息损失 → Phase A **暂未支持**（±1 日设定下）。
- **仍成立的价值**：Rule / Leakage / Fair / Proposed **解耦实验框架**；相对 Teacher 的可比评测；生命周期旁证。
- **Phase B 目标调整**：由「证明 Proposed 一定更好」转为 **验证 Phase A 负结果在更大 train 下是否仍成立**，并消融 **triplet 时间间隔**（±1 vs ±3 vs ±5）。

---

## 5. 给导师/答辩的一句话（推荐）

> Phase A 烟测中，连续三时刻 ADT（±1 日）未优于单时刻 ADT；Proposed 召回略升但精度与 mAP 略降。在 OW(P24) 弱监督与 90 帧短训下，单帧 ADT 已能提供主要涡旋结构信息，额外时间信息尚未转化为净识别增益。该结论为趋势性判断，正式结论待 Phase B 扩 train 与时间间隔消融。

---

## 6. Phase B 建议路线（待执行）

### 6.1 B0（优先，同 NC 内扩日历）

| 实验 ID | 输入 | train | epoch | 目的 |
| --- | --- | --- | ---: | --- |
| Fair-B1 | ADT(t)×3 | **2018 全年** | 50→100 | 新 Fair 基线 |
| Proposed-B1 | ADT(t−1,t,t+1) | 2018 全年 | 50→100 | ±1 在足量数据下是否仍负 |
| **Proposed-B2** | **ADT(t−3,t,t+3)** | 2018 全年 | 50→100 | **最小关键消融**：间隔太短 vs 时序无用 |
| Leakage-B1 | ADT+U+V | 2018 全年 | 50→100 | U/V 旁证是否随数据量改善 |

时间紧时的**最小对打**：Fair-B1 vs Proposed-B2。

### 6.2 B1（论文正式，v6 §7.4）

- train：多个**连续季度/年块**，覆盖 train 三 NC（1993–2022 池）
- val：2024 全年（同 Phase A）
- test：**2023 全年终评一次**
- epoch：100 或 50 + 收敛曲线

### 6.3 实现前置

- `eddy_yolo_export` 增加 `--triplet-offset`（1/3/5），全文一致更新 v6.md
- 仍共用 OW(P24) labels，仅改 `images/`

---

## 7. 复现命令

```powershell
Set-Location F:\创赛

.\scripts\run_eddy_v6_phase_a_export.ps1

python scripts/eddy_compare_ow_rule_vs_yolo.py `
  --dataset-root AutoDL/dataset/eddy_v6_fair --splits val `
  --single-percentile 24 --skip-yolo --v6-teacher `
  --out-md submission/tables/eddy_v6_rule_compare.md

python -m src.eddy.train --config config/eddy_v6_fair.yaml
python -m src.eddy.train --config config/eddy_v6_proposed.yaml
python -m src.eddy.train --config config/eddy_v6_leakage.yaml
python -m src.eddy.eval --config config/eddy_v6_fair.yaml --splits val
python -m src.eddy.eval --config config/eddy_v6_proposed.yaml --splits val
python -m src.eddy.eval --config config/eddy_v6_leakage.yaml --splits val
python scripts/eddy_write_v6_fair_vs_proposed.py
python scripts/eddy_v6_lifetime_stats.py
```

---

## 8. 不可破坏项

- 未修改 `eddy_cloud_fair` 默认 config / 导出 / 训练入口
- Phase A train **锁死** 2018Q1；历史复现必须仍可跑通
- eval JSON 字段与 `src.eddy.eval` 约定一致
- Rule **不参与** Fair vs Proposed 排名
