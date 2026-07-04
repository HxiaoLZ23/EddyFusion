# V6 Phase B0 — 2018 全年 + offset 消融（355/356 stem）

> 独立实验线；不替换 `eddy_cloud_fair`。规范见 `开发阶段文档/基于时空一致性弱监督标签的中尺度涡旋实例分割训练方案v6.md`。

## 硬锁

| 项 | 设定 |
| --- | --- |
| train | 2018-01-01~12-31，`k_max=5` → **355** 中心日（2018-01-06~12-26） |
| val | 2024 闰年，`k_max=5` 仅裁剪（**无** Phase A `skip_boundary_days`）→ **356** |
| 标签 | OW P24；ADT 归一化方案 A |
| stem | 五 mode 完全一致；manifest：`submission/tables/eddy_v6_b0_stem_manifest.json` |

## 实验 ID

| ID | input-mode | triplet-offset | 优先级 |
| --- | --- | --- | --- |
| Fair-B0 | fair | — | P0 |
| Proposed-B0-k3 | triplet | 3 | P0 |
| Proposed-B0-k1 | triplet | 1 | P1 |
| Proposed-B0-k5 | triplet | 5 | P2 |
| Leakage-B0 | leakage | — | P2 |

## 复现命令

```powershell
# 导出 + stem 验收
powershell -ExecutionPolicy Bypass -File scripts/run_eddy_v6_phase_b0_export.ps1

# P0 训练
powershell -ExecutionPolicy Bypass -File scripts/run_eddy_v6_b0_train.ps1 -Priority p0

# P0 eval + 主表
powershell -ExecutionPolicy Bypass -File scripts/run_eddy_v6_b0_eval.ps1 -Priority p0

# Rule 旁证（同 B0 stem）
python scripts/eddy_compare_ow_rule_vs_yolo.py --dataset-root AutoDL/dataset/eddy_v6_b0_leakage --skip-yolo --v6-teacher --single-percentile 24 --splits val --out-json submission/tables/eddy_v6_b0_rule_compare.json --out-md submission/tables/eddy_v6_b0_rule_compare.md
```

## Phase A 对照（烟测 90/364）

Fair 0.705 > Proposed±1 0.680 > Leakage 0.641。**B0 目标**：足量 train + offset 消融下验证负结果是否仍成立。

## 结果（待填）

主表：`submission/tables/eddy_v6_b0_fair_vs_proposed.md`

| 组别 | mAP50 | P | R |
| --- | ---: | ---: | ---: |
| Fair-B0 | 0.823 | 0.788 | 0.728 |
| Proposed-B0-k3 | 0.766 | 0.693 | 0.712 |
| Proposed-B0-k1 | — | — | — |

**Proposed-k3 − Fair**：ΔmAP50=−0.057；ΔP=−0.096；ΔR=−0.017（50 ep 本地，355 train / 356 val）

## 数据集路径

- `AutoDL/dataset/eddy_v6_b0_{fair,proposed_k1,proposed_k3,proposed_k5,leakage}`
- Config：`config/eddy_v6_b0_*.yaml`
- 权重：`outputs/eddy_v6_b0_*/`
