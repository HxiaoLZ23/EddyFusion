# 模块 C（风浪异常）回归指标汇总（val / test）

| 指标 | val | test |
|------|-----|------|
| MAE 风速 | 4.430597 | 0.591457 |
| MAE 波高 | 5.863316 | 0.345821 |
| MAE 平均 | 5.146956 | 0.468639 |
| RMSE 风速 | 5.473613 | 0.786927 |
| RMSE 波高 | 7.020177 | 0.479556 |
| RMSE 平均 | 6.246895 | 0.633242 |

## 复现与口径

- **权重（材料口径）**：`outputs/anomaly/best.pt`（新键名 `wind_head` / `wave_head`）
- **评估命令**：
  - `python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split val`
  - `python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split test`
- **JSON 来源**：`F:/创赛/outputs/anomaly/metrics_summary_val.json`、`F:/创赛/outputs/anomaly/metrics_summary_test.json`（由 `eval` 自动写出）
- **旧格式 checkpoint**：仅作加载兼容自检时 val MAE 可能极大，见 `docs/后续开发工作清单_未完成项与云端L0专项.md` §1.3.1。

> **提示**：当前 JSON 中 **val** 与 **test** 的 `mae_avg` 差距较大；请确认两个 JSON 均由同一 `best.pt`（新键名双头）生成。若为兼容自检用的旧格式权重，见工作清单 §1.3.1。
