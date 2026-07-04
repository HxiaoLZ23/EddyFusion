# 涡旋通道消融：以 3ch 为基线的增通道实验

> **方法论（与赛题叙事对齐）**  
> 本项目**不必**证明「7ch 优于 3ch」。正确做法是：在 **3ch 伪彩基线**上，逐路或组合增加物理派生通道，用同一伪标签与公平 eval **测量**哪些通道对当前任务（OW 投票弱监督下的实例分割）有提升。  
> **不做**：先固定 7ch 再写「7ch 更科学 / 无 Mask」的主结论。  
> **要做**：实验 **确定** 通道组合；7ch full 仅是「ζ+OW+Grad 全开」的一种汇总行。

代码：`src/eddy/stacked_physics.py` → `ABLATION_PROFILES`、`build_physics_stacked_ablation`。

---

## 1. 实验矩阵（3ch + Δ）

前 3 通道恒为 BGR 伪彩（adt / ugos / vgos 分位归一化），与 `AutoDL/dataset/eddy` 的 3ch 导出一致。附加通道均由同一帧的 u,v 与 adt 计算。

| 优先级 | profile | 通道数 | 相对 3ch 新增 | 数据集目录 |
|--------|---------|--------|---------------|------------|
| **基线** | （3ch） | 3 | — | `AutoDL/dataset/eddy` |
| 单路 | `4_bgr_zeta` | 4 | +ζ | `eddy_ablation/4_bgr_zeta` |
| 单路 | `4_bgr_ow` | 4 | +OW | `eddy_ablation/4_bgr_ow` |
| 单路 | `5_bgr_grad` | 5 | +GradX/Y | `eddy_ablation/5_bgr_grad` |
| 双路 | `5_no_grad` | 5 | +ζ+OW | `eddy_ablation/5_no_grad` |
| 双路 | `6_no_ow` | 6 | +ζ+Grad | `eddy_ablation/6_no_ow` |
| 双路 | `6_no_zeta` | 6 | +OW+Grad | `eddy_ablation/6_no_zeta` |
| 全开 | `7` / 7ch fair | 7 | +ζ+OW+Grad | `eddy_enh7` |

**解读方式**：每一行与 **3ch 基线** 比 val/test `mask_map50`；单路行回答「这一路是否值得加」；双路/全开回答组合效应。若 7ch ≤ 3ch，应写「在当前伪标签口径下，增通道未带来可测提升」，而非硬证 7ch。

**共同约定**：

- 同一 OW 伪标签（`vote-percentiles 12,18,24,30`，`vote-min 2`）。
- 同一 `time_stride`（公平对比 **7**，与 `eddy_cloud_fair` 一致）。
- 多通道训练：`init_from_baseline: outputs/eddy_cloud_fair/last.pt`（首层扩展，与 3ch 权重对齐）。
- 指标：`python -m src.eddy.eval --splits val,test` → `mask_map50`；可选 E5 pred vs vote mask IoU。

---

## 2. 与「从 7ch 往下删」的关系

早期矩阵以 **7ch full 为中心**做「去掉某路」（`6_no_ow` 等），命名仍保留，但在论文表中应 **重排为 3ch 增通道叙事**：

| 旧视角（7ch-centric） | 新视角（3ch-centric） |
|----------------------|----------------------|
| 7ch 去掉 OW | 3ch + ζ + Grad |
| 7ch 去掉 Grad | 3ch + ζ + OW |
| 7ch 去掉 ζ | 3ch + OW + Grad |
| 仅 BGR+ζ | 3ch + ζ |

---

## 3. 要回答的问题

| 对比 | 若 mAP 相对 3ch | 解读 |
|------|-----------------|------|
| `4_bgr_zeta` | ↑ / ≈ / ↓ | 显式 ζ 是否有边际收益 |
| `4_bgr_ow` | ↑ / ≈ / ↓ | 显式 OW（与伪标签同源量）是否帮助或冗余 |
| `5_bgr_grad` | ↑ / ≈ / ↓ | ADT 梯度两路是否帮助 |
| 双路与 7ch | 相对单路 | 是否存在协同；全开是否优于最佳子集 |

**已有公平 eval 提示（非消融 Full）**：3ch val/test mAP@0.5 ≈ 0.934/0.915；7ch fair ≈ 0.834/0.850——支持「先验不假定 7ch 更优」，应用本矩阵 **实测** 哪几路有贡献。

---

## 4. 本地一条龙

```powershell
Set-Location F:\创赛

# 烟测（导出 2 帧/文件 + 每模型 5 epoch）
.\scripts\run_eddy_7ch_ablation_local.ps1 -Smoke

# 正式（全量导出 + 100 epoch，耗时 ×6 profile）
.\scripts\run_eddy_7ch_ablation_local.ps1 -Full

python scripts/eddy_write_ablation_map_table.py
```

产物：`submission/tables/eddy_ablation_7ch_matrix.md`（表头已改为 3ch 增通道表述）

---

## 5. 单步命令（示例：3ch + OW）

```powershell
python -m src.preprocess.eddy_yolo_export `
  --data-config config/data.yaml `
  --out AutoDL/dataset/eddy_ablation/4_bgr_ow `
  --stack-physics-npy --stack-profile 4_bgr_ow `
  --time-stride 7

python -u -m src.eddy.train --config config/eddy_ablation/4_bgr_ow.yaml
python -m src.eddy.eval --config config/eddy_ablation/4_bgr_ow.yaml `
  --ckpt outputs/eddy_ablation/4_bgr_ow/best.pt --splits val,test
```

---

## 6. 论文推荐表述

在 OW 多分位投票伪标签弱监督下，我们以 **3ch 海面动力伪彩**为基线，对 ζ、OW、ADT 梯度等派生通道做 **增通道消融**，以 val/test 实例分割 mAP 与（可选）预测–投票 mask 重叠度 **选择**输入组合。当前全量公平对比显示 3ch 高于 7ch full 时，应如实报告并讨论伪标签–输入信息重叠与过拟合风险，**而非**将 7ch 标为默认最优架构。

---

## 7. 与 E0–E7 实验包

- E0–E5（`涡旋_无人工标注实验报告.md`）：伪标签敏感性、标签破坏、vote IoU。
- 本矩阵：**通道选择**主证据；完成后更新该报告 **§8 E6** 与主表。

---

## 8. 需人工/云端

- 全量 train × **6** profile × 100 epoch：本地 6GB 显存建议 `batch_size: 4`，可夜间或 AutoDL。
- 7ch full 若需与消融完全同 init 策略，可另增一行 `init_from_baseline` 重训（可选）。

---

## 9. 完成状态（2026-06-04）

| 项 | 状态 |
|----|------|
| 单路 `4_bgr_zeta` / `4_bgr_ow` / `5_bgr_grad` | ✅ Full 100 epoch + eval |
| 双路三组 | ⏭️ 动态规划跳过（单路均 harmful） |
| 7ch 全开 | ✅ 使用既有 `eddy_enh7_cloud_fair`（未重训） |
| E6 表 | ✅ `submission/tables/eddy_ablation_E6_channel_ablation.md` |
| 实验报告 §8 | ✅ `涡旋_无人工标注实验报告.md` |
| 结论 | **主方案维持 3ch**；增通道无 mAP 提升 |

动态规划快照：`.cursor/goal-abl-dynamic.json`（`phase: done`）。
