# V6 Phase A — OW(P24) Teacher 上界旁证

> **Teacher 上界**：OW(P24) 规则链重放伪标签；P/R≈1 为标签生成上界。
> **不参与 Fair vs Proposed 排名**；主表见 `submission/tables/eddy_v6_fair_vs_proposed.md`。

- 数据集：`F:\创赛\AutoDL\dataset\eddy_v6_fair`
- NC：`F:\创赛\服创数据集\中尺度涡识别`
- YOLO 权重：`None`
- splits：val
- conf：0.25

## 1. 相对伪标签（instance IoU≥0.5）

| split | 方法 | n | P | R | F1 | matched mIoU | 小涡旋召回 | 边界粗糙度↓ | 边界梯度↓ | 粘连/图 | 过分割/图 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | ow_rule | 364 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.52 | 0.0754 | 0.00 | 0.00 |

## 2. 时序稳定性（相邻帧 union IoU，越高越稳）

| split | 方法 | 配对帧数 | mean union IoU | mean |Δ实例数| |
| --- | --- | ---: | ---: | ---: |
| val | ow_rule | 363 | 0.681 | 1.27 |

## 3. OW 超参扰动敏感性（仅 OW 规则；std 越小越稳）

| split | OW 实例数 std/图 | OW 面积占比 std/图 |
| --- | ---: | ---: |
| val | 2.737 | 0.0080 |

## 4. 单帧推理耗时

- OW 规则（CPU）：86.6 ms/帧
- YOLO（skipped）：0.0 ms/帧

## 5. 答辩要点（自动生成）

**val Teacher（OW P24）**：相对伪标签 P/R=1.000/1.000，matched mIoU=1.000。 **用途**：Teacher 上界旁证；**不参与** V6 Fair vs Proposed 主排名。主结论以 `eddy_v6_fair_vs_proposed.md` 中 Proposed−Fair 的 ΔmAP/ΔP/ΔR 为准。

明细 CSV：`F:\创赛\submission\tables\eddy_v6_rule_compare.csv`
JSON：`F:\创赛\submission\tables\eddy_v6_rule_compare.json`
