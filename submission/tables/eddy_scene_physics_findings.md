# 7ch 无 Mask 补充分析结论（无人工真值）

数据口径：`AutoDL/dataset/eddy` 与 `AutoDL/dataset/eddy_enh7`，train/val/test = 1566/53/53；伪标签作为弱参考，物理一致性指标作为补充评价。注意：本文数值来自本次统一数据重评估，与早期归档的 `0.762/0.730` 不是同一可直接混用口径。

## 主要结论

- **总体伪标签拟合能力仍是 3ch 更强**：本次用 `outputs/eddy_cloud_fair/last.pt` 与 `outputs/eddy_enh7_cloud_fair/best.pt` 在同一 `AutoDL/dataset` 上重评估，3ch val/test mask mAP@0.5 为 0.934/0.915，7ch 为 0.834/0.850。
- **高置信度工作点下 7ch 更保守、误检更少**：在 `conf=0.25` 的弱参考匹配中，7ch 的 FP/image 明显低于 3ch（val: 0.226 vs 0.962；test: 0.170 vs 1.321），但召回显著下降。
- **物理一致性没有支撑“7ch 全面更物理”**：OW 低值占比、涡度符号一致性、边界梯度强度整体仍由 3ch 更高或相近。由于伪标签和 3ch 输入同源，这类一致性也会强化 3ch 的优势；因此当前产物不宜写成 7ch 在物理一致性上全面优于 3ch。
- **降低 7ch 阈值可提高召回，但优势变弱**：`3ch conf=0.25 / 7ch conf=0.05` 时，7ch recall 提升到 val/test 0.246/0.263，但 precision 与 FP/image 变差，说明 7ch 主要差异是置信度与选择性，而不是稳定反超。
- **IoU 解读需区分两种口径**：`mean_iou` 含漏检惩罚，TP=0 的图按 0 计；`matched_mean_iou` 只看已匹配目标的形状质量，更适合讨论边界贴合度。

## 推荐论文表述

当前结果更适合支持如下口径（与 `docs/实验与结果归档/涡旋_无人工标注实验报告.md` E1–E5 一致）：

> 在 OW 多分位投票伪标签口径下，3ch 伪彩基线 mask mAP@0.5 高于 7ch 显式物理通道；标签打乱试验表明评测与伪标签结构绑定，且 3ch 预测与 OW 投票掩膜的空间重合度更高。7ch 与 3ch **输入侧均无 Mask**；7ch 的差异主要体现在高置信度下更低误检、更低召回，以及待完成的通道消融（如去掉 OW 的 5ch）。不宜将 7ch 表述为已证实的「无 Mask 输入故更优」主方案。

## 产物

- 标准阈值表：`submission/tables/eddy_scene_physics_cloud_fair.groups.md`
- 校准阈值表：`submission/tables/eddy_scene_physics_cloud_fair_calibrated.groups.md`
- 标准阈值图：`submission/figures/eddy_7ch_advantage_analysis.png`
- 校准阈值图：`submission/figures/eddy_7ch_advantage_analysis_calibrated.png`
