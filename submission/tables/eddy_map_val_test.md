# 涡旋分割验证指标（Ultralytics YOLO-seg，固定划分）

列名前缀：`val_` / `test_`；指标键与 `metrics_summary_*.json` 内 `metrics` 一致。

| 实验 | split | mAP50 | mAP50-95 | mAP75 | mean P | mean R | mean F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3ch RGB 基线 | val | 0.761799 |  |  |  |  |  |
| 3ch RGB 基线 | test | 0.730303 |  |  |  |  |  |
| 8ch 物理增强 | val | 0.838573 |  |  |  |  |  |
| 8ch 物理增强 | test | 0.825861 |  |  |  |  |  |

说明：``mask_mean_precision`` / ``mask_mean_recall`` / ``mask_mean_f1`` 为 Ultralytics 对各类别指标的均值；推理侧频域/unsharp、多尺度 TTA 若未烘焙进验证集，`scripts/eddy_inference_ablate.py` 仅反映检测计数/置信度灵敏度；正式分割评测口径以本表 `eval` 为准。