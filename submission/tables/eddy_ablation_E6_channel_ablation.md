# E6 3ch 增通道消融（Full，2026-06-04）

> 方法论：在 **3ch 伪彩基线**上逐路增通道，用实验选通道，非先定 7ch。  
> 计划：`docs/实验与结果归档/涡旋_7ch通道消融计划.md`  
> 动态规划：`.cursor/goal-abl-dynamic.json`、`submission/tables/eddy_ablation_dynamic_plan.md`

## 训练约定

| 项 | 值 |
|----|-----|
| 伪标签 | OW 投票 12/18/24/30，`vote_min=2` |
| 划分 | 与 `eddy_cloud_fair` 一致，`time_stride=7` |
| 初始化 | `init_from_baseline: outputs/eddy_cloud_fair/last.pt` |
| 训练 | 100 epoch，`batch_size=4`，`yolov8n-seg` |
| 指标 | `mask_map50`（val/test） |

## 主结果

| 实验 | profile | ch | val mAP@0.5 | test mAP@0.5 | Δval vs 3ch | 判定 |
|------|---------|----|-------------|--------------|-------------|------|
| **3ch 基线** | — | 3 | **0.934** | **0.915** | 0 | baseline |
| +ζ | `4_bgr_zeta` | 4 | 0.775 | 0.770 | −0.158 | harmful |
| +OW | `4_bgr_ow` | 4 | 0.842 | 0.839 | −0.092 | harmful |
| +Grad | `5_bgr_grad` | 5 | 0.831 | 0.839 | −0.102 | harmful |
| +all（先验） | 7ch fair | 7 | 0.834 | 0.850 | −0.100 | harmful |

阈值：Δval ≥ +0.005 → helpful；Δval ≤ −0.02 → harmful。

## 双路组合（动态规划跳过）

三路单通道均为 harmful，未再训练以下 profile（节省算力，符合「实验确定通道」）：

| profile | 相对 3ch | 跳过原因 |
|---------|----------|----------|
| `5_no_grad` | +ζ+OW | ζ、OW 单路均有害 |
| `6_no_ow` | +ζ+Grad | ζ、Grad 单路均有害 |
| `6_no_zeta` | +OW+Grad | OW、Grad 单路均有害 |

## 结论（E6）

1. **主交付输入应维持 3ch**；在当前 OW 伪标签弱监督下，显式叠加 ζ / OW / ADT 梯度均未带来 mAP 提升。
2. **单路最好为 +OW**（val 0.842），仍显著低于 3ch；**+ζ 最差**（0.775）。
3. 与 E1（7ch fair 0.834/0.850）一致；**不宜**以「7ch 无 Mask 更科学」替代 3ch。
4. 双路组合无必要复跑（规划器 `phase: done`）。

## 复现

```powershell
Set-Location F:\创赛
python scripts/eddy_ablation_dynamic.py init
.\scripts\run_eddy_ablation_goal_step.ps1   # 按 next 逐 profile
# 或单 profile：
.\scripts\run_eddy_7ch_ablation_local.ps1 -Full -Profile 4_bgr_ow
```

权重：`outputs/eddy_ablation/{4_bgr_zeta,4_bgr_ow,5_bgr_grad}/best.pt`  
汇总：`submission/tables/eddy_ablation_7ch_matrix.md`
