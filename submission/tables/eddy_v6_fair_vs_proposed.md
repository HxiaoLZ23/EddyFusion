# V6 Phase A — Fair vs Proposed（val 2024，50 epoch 烟测）

> **主比较**：Fair（单帧 ADT×3）vs Proposed（三时刻 ADT）。
> Leakage（ADT+U/V）与 Rule（OW P24 重放）为旁证，**不参与排名**。

训练：2018Q1（90 帧）；评测：2024 val（364 帧）；标签 OW(P24)；归一化方案 A。

| 组别 | 角色 | 参与排名 | mAP50 | mAP50-95 | P | R | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fair | 新基线 | ✓ 主比 | 0.705 | 0.323 | 0.690 | 0.619 | 0.652 |
| Proposed | 新方案（三时刻 ADT） | ✓ 主比 | 0.680 | 0.304 | 0.628 | 0.646 | 0.636 |
| Leakage | U/V 旁证 | ✗ | 0.641 | 0.264 | 0.617 | 0.617 | 0.614 |
| Rule (OW P24) | Teacher 上界 | ✗ | 1.000 | — | 1.000 | 1.000 | 1.000 |

## 主结论（Proposed − Fair）

ΔmAP50=-0.025；ΔP=-0.062；ΔR=+0.027

## 脚注

- **Rule**：相对伪标签 instance IoU≥0.5 重放；P/R≈1 为 Teacher 上界；mAP 列用 matched mIoU 示意，非 YOLO mAP。
- **Leakage**：含 U/V 伪彩输入，仅作信息泄漏旁证，不得与 Fair/Proposed 直接比优劣。
- 指标来自 `python -m src.eddy.eval --splits val` 的 `metrics_summary_val.json`。
