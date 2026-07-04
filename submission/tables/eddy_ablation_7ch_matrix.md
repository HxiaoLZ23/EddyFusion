# 3ch 增通道消融 mAP 汇总（相对 3ch 基线）

| 实验 | val mask mAP@0.5 | test mask mAP@0.5 | 备注 |
| --- | --- | --- | --- |
| 3ch (baseline) | 0.933831 | 0.914533 |  |
| +zeta (4_bgr_zeta) | 0.77541 | 0.770109 |  |
| +ow (4_bgr_ow) | 0.841635 | 0.83877 |  |
| +grad (5_bgr_grad) | 0.831476 | 0.839393 |  |
| +zeta+ow (5_no_grad) | — | — | 动态规划跳过（单路均 harmful） |
| +zeta+grad (6_no_ow) | — | — | 同上 |
| +ow+grad (6_no_zeta) | — | — | 同上 |
| +all (7ch) | 0.834321 | 0.849803 | 先验 fair 权重，未重训 |

> Full 单路消融见 `submission/tables/eddy_ablation_E6_channel_ablation.md`。
