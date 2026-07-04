# 模块 C：24h 超前回归指标（自回归 rollout）

**数据来源**：本地 **2026-06-05** 对 `outputs/anomaly/best.pt` 执行 `anomaly_eval_horizon24.py` 后归档；与 [AutoDL_outputs_云端结果归档.md](../实验与结果归档/AutoDL_outputs_云端结果归档.md) §6.5 同步。

**权重**：`outputs/anomaly/best.pt`（训练目标为 **3h 一步**，非 24h 直接监督）  
**评估**：`horizon_steps=8`（8×3h=24h）自回归 rollout；持续性 = 窗口末值外推 24h  
**量纲**：|U10| m/s、SWH m，无 StandardScaler  

复现：

```bash
python scripts/anomaly_eval_horizon24.py --split both --horizon-hours 24
```

| 指标 | val | test |
|------|-----|------|
| MAE 风速 | 0.2429 | 0.2480 |
| MAE 波高 | 0.0706 | 0.0730 |
| MAE 平均 | **0.1567** | **0.1605** |
| RMSE 风速 | 0.3073 | 0.3183 |
| RMSE 波高 | 0.0936 | 0.0968 |
| RMSE 平均 | **0.2005** | **0.2075** |
| 持续性 MAE 平均 | 0.1743 | 0.1854 |
| MAE/持续性 | 0.899 | 0.866 |

## 与 3h 一步指标对照

| 超前 | val MAE 平均 | test MAE 平均 | 说明 |
|------|-------------|---------------|------|
| **3h**（训练口径） | 0.0295 | 0.0292 | 区域平均 + 短超前，持续性 ~0.063 |
| **24h**（rollout） | 0.1567 | 0.1605 | 更接近导师直觉；略优于持续性 |

JSON：`outputs/anomaly/metrics_summary_{val,test}_horizon24h.json`
