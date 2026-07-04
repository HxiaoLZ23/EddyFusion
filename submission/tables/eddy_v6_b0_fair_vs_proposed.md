# V6 Phase B0 — Fair vs Proposed offset 消融（val 2024，50 epoch）

> **主比较**：Fair vs Proposed-B0-k3（关键消融）；k1/k5 为 offset 对照。
> Leakage 与 Rule 为旁证，**不参与排名**。

训练：2018 全年 k_max=5 交集（manifest 355/356 stems）；标签 OW(P24)；归一化方案 A。

| 组别 | 角色 | 参与排名 | mAP50 | mAP50-95 | P | R | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fair-B0 | 主基线 | ✓ 主比 | 0.823 | 0.469 | 0.788 | 0.728 | 0.756 |
| Proposed-B0-k3 | 关键消融 offset=3 | ✓ 主比 | 0.766 | 0.402 | 0.693 | 0.712 | 0.701 |
| Proposed-B0-k1 | 对照 offset=1 | ✓ 主比 | 0.820 | 0.470 | 0.757 | 0.750 | 0.749 |
| Proposed-B0-k5 | 边界 offset=5 | ✗ | 0.735 | 0.361 | 0.670 | 0.685 | 0.674 |
| Leakage-B0 | U/V 旁证 | ✗ | 0.814 | 0.433 | 0.740 | 0.741 | 0.739 |
| Rule (OW P24) | Teacher 上界 | ✗ | — | — | — | — | — |

## 主结论（Proposed-B0-k3 − Fair-B0）

ΔmAP50=-0.057；ΔP=-0.096；ΔR=-0.017

## offset 对照（Proposed-B0-k1 − Fair-B0）

ΔmAP50=-0.003；ΔP=-0.031；ΔR=+0.022

## 脚注

- stem 清单见 `submission/tables/eddy_v6_b0_stem_manifest.json`。
- val 仅用 k_max=5 裁剪，**不**叠加 Phase A 的 skip_boundary_days。
- Rule：同 B0 stem 子集；`--v6-teacher` 重放 OW P24。
