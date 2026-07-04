# V6 Fair-B0 — val / test 指标（mask，Ultralytics YOLO-seg）

> **test**：2023 全年 `20230101_20231231`，355 帧，k_max=5 与 train/val 同 stem 口径。  
> **val**：2024 全年 `20240101_20241231`，356 帧。  
> 权重：`outputs/eddy_v6_b0_fair/best.pt`；config：`config/eddy_v6_b0_fair.yaml`。

| Split | 帧数 | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 356 | **0.824** | 0.475 | **0.788** | **0.728** | **0.756** |
| test | 355 | **0.801** | 0.448 | **0.772** | **0.714** | **0.742** |

复现：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_eddy_v6_b0_export_test.ps1
```

或仅 eval（test 已导出时）：

```bash
python -m src.eddy.eval --config config/eddy_v6_b0_fair.yaml --ckpt outputs/eddy_v6_b0_fair/best.pt --splits test
```

产物：`outputs/eddy_v6_b0_fair/metrics_summary_test.json`
