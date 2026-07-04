# 模块 C（风浪异常）回归指标汇总（val / test）

**数据来源**：本地 **2026-05-20** 对 `outputs/anomaly/best.pt` 执行 split `eval` 后由 `scripts/anomaly_export_metrics_table.py` 导出。

| 指标 | val | test |
|------|-----|------|
| MAE 风速 | 0.050069 | 0.049546 |
| MAE 波高 | 0.008946 | 0.008770 |
| MAE 平均 | **0.029508** | **0.029158** |
| RMSE 风速 | 0.064351 | 0.062827 |
| RMSE 波高 | 0.011265 | 0.011022 |
| RMSE 平均 | **0.037808** | **0.036924** |
| `passed`（`mae_avg` &lt; 0.5） | true | true |

## 复现与口径

- **权重（材料口径）**：`outputs/anomaly/best.pt`（键名 `wind_head` / `wave_head`）
- **评估命令**：
  - `python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split val`
  - `python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split test`
- **JSON 来源**：`outputs/anomaly/metrics_summary_val.json`、`outputs/anomaly/metrics_summary_test.json`（由 `eval` 自动写出）
- **勿作业务口径**：`metrics_summary.json`（无 split）或旧格式 `head.*` 权重自检；若见 val MAE 风速 ~4+，见归档 §6 «解读要点»。
