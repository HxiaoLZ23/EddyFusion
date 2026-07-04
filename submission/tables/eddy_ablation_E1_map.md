# 涡旋分割验证指标（Ultralytics YOLO-seg，固定划分）

列名前缀：`val_` / `test_`；指标键与 `metrics_summary_*.json` 内 `metrics` 一致。

| 实验 | split | mAP50 | mAP50-95 | mAP75 | mean P | mean R | mean F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3ch_cloud_fair | val | 0.933831 | 0.605525 | 0.722942 | 0.896512 | 0.86482 | 0.880275 |
| 3ch_cloud_fair | test | 0.914533 | 0.58735 | 0.681606 | 0.853587 | 0.842942 | 0.848199 |
| 7ch_cloud_fair | val | 0.834321 | 0.490296 | 0.55948 | 0.817621 | 0.742014 | 0.772289 |
| 7ch_cloud_fair | test | 0.849803 | 0.507329 | 0.579285 | 0.814003 | 0.73533 | 0.765908 |

说明：``mask_mean_precision`` / ``mask_mean_recall`` / ``mask_mean_f1`` 为 Ultralytics 对各类别指标的均值；推理侧频域/unsharp、多尺度 TTA 若未烘焙进验证集，`scripts/eddy_inference_ablate.py` 仅反映检测计数/置信度灵敏度；正式分割评测口径以本表 `eval` 为准。