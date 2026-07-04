# 涡旋：OW 规则法 vs YOLOv8-seg 对比实验（第 6 章答辩）

> 生成：2026-06-05。仓库根：`创赛/`。  
> **回应质疑**：「推理时也有 U、V，为什么不直接 OW→投票→连通域？YOLO 是否只是复现伪标签？」

---

## 1. 实验设计

| 项 | 说明 |
|----|------|
| **OW 规则链** | U/V → Okubo–Weiss → 多分位投票 (12/18/24/30, ≥2 票) → 连通域 → 多边形；参数与 `eddy_yolo_export` 伪标签导出 **完全一致** |
| **YOLO** | 3ch 伪彩 RGB → `yolov8n-seg`，权重 `outputs/eddy_cloud_fair/last.pt`，conf=0.25 |
| **弱参考** | OW 导出伪标签（**非** META/人工 polygon） |
| **数据** | val/test 各 53 帧，`AutoDL/dataset/eddy`；NC `服创数据集/中尺度涡识别` |
| **脚本** | `python scripts/eddy_compare_ow_rule_vs_yolo.py --splits val,test` |

### 指标分组

1. **相对伪标签**：P/R/F1、matched mIoU、小涡旋召回、边界粗糙度、粘连/过分割  
2. **YOLO vs OW 直接输出**：以 OW 规则实时输出为参考，不经过 `.txt` 伪标签文件  
3. **时序稳定性**：同 NC 序列相邻帧 union mask IoU  
4. **OW 超参扰动**：4 组 vote 配置下实例数/面积波动（YOLO 不经过此链）  
5. **耗时**：单帧 CPU OW vs GPU YOLO  

---

## 2. 主结果

表文件：`submission/tables/eddy_ow_rule_vs_yolo_compare.md`  
图：`submission/figures/eddy_ow_rule_vs_yolo_compare.png`

### 2.1 相对伪标签

| split | 方法 | P | R | F1 | matched mIoU | 边界粗糙度 | 过分割/图 |
|-------|------|---:|---:|---:|---:|---:|---:|
| val | **OW 规则** | **1.000** | **1.000** | **1.000** | **1.000** | 5.56 | 0.00 |
| val | YOLO | 0.883 | 0.692 | 0.776 | 0.763 | **5.29** | 0.32 |
| test | **OW 规则** | **1.000** | **1.000** | **1.000** | **1.000** | 5.57 | 0.00 |
| test | YOLO | 0.852 | 0.707 | 0.773 | 0.762 | **5.36** | 0.47 |

**解读（必写进论文）**：

- OW 规则对伪标签 **完美复现**（P/R=1.0）——因伪标签即同一 OW 链导出，这是 **预期结果**，不是 OW「更强」的独立证据。  
- YOLO val/test mAP≈0.93/0.92 与上表 YOLO P/R≈0.88/0.85、R≈0.69/0.71 **一致**：高 mAP 主要度量 **对 OW 规则的拟合**，而非超越 OW。  
- YOLO 在伪标签口径上 **召回低于 OW 重放**（小涡旋召回 OW=1.0 vs YOLO≈0.34–0.42），存在 **过分割**（test 0.47/图）。

### 2.2 YOLO vs OW 直接输出

| split | P | R | matched mIoU |
|-------|---:|---:|---:|
| val | 0.883 | 0.692 | 0.763 |
| test | 0.852 | 0.707 | 0.762 |

与 §2.1 中 YOLO 相对伪标签一致 → 伪标签文件与 OW 实时输出在本数据集上 **等价**。

### 2.3 工程向差异（YOLO 可主张的点）

| 维度 | OW 规则 | YOLO | 结论 |
|------|---------|------|------|
| 边界粗糙度 (↓更好) | 5.56 / 5.57 | **5.29 / 5.36** | YOLO 边界略平滑（约 **4–5%**） |
| 时序 union IoU (↑更好) | 0.273 / 0.253 | **0.296 / 0.277** | YOLO 相邻帧略稳（约 **+8% 相对**） |
| OW 超参扰动实例数 std/图 | **2.72 / 2.98** | 0（不依赖 OW 链） | YOLO **免疫** 投票阈值扰动 |
| 单帧耗时 | ~43 ms CPU | ~47 ms GPU | 同量级；YOLO 可 batch 部署 |
| 过分割 | 0 | 0.32–0.47/图 | OW 更「整」；YOLO 可通过 conf↑ 换 precision |

---

## 3. 答辩话术（建议）

### Q：为什么不直接跑 OW？

**A（诚实版）**：

> 可以。在本实验弱参考下，推理期 OW 规则链 **100% 复现** 训练伪标签；若评价标准就是「与 OW 伪标签一致」，规则法上限更高。  
> 我们采用 YOLO 的原因不是「神经网络 magically 比物理公式更准」，而是：  
> 1. **规则蒸馏**：将 OW→投票→连通域 压缩为 **RGB 前向模型**，便于与 React 前端、GPU 批推理、三模块统一栈集成；  
> 2. **阈值链免疫**：OW 对 vote 超参敏感（扰动下实例数 std≈3/图，见 E3）；YOLO 推理 **不经过** 该链条；  
> 3. **输出正则化**：边界略平滑、时序 union IoU 略高，适合可视化与下游统计；  
> 4. **扩展性**：若未来引入人工标注或外部 META 产品，可在同一检测头上 **微调**，而无需重写规则参数。  
> 93% mAP 应表述为 **「对 OW Teacher 的 Student 拟合度」**，而非独立真值上的 SOTA。

### Q：是不是「用神经网络复现了一遍自己生成的标签」？

**A**：

> **在伪标签口径上， largely yes**——这也是本实验 deliberately 暴露的结论。  
> 论文贡献应调整为：**弱监督下 OW Teacher → YOLO Student 的蒸馏框架 + 工程落地**，并报告蒸馏保真度（~76% matched mIoU / 93% mAP）与规则法差异。  
> 若要 claim「YOLO 优于 OW」，必须补充 **独立人工样例** 或 **输入扰动（NC 噪声、缺通道）鲁棒性** 实验——当前数据 **不支持** 「YOLO 小涡旋召回更高」（反而更低）。

---

## 4. 第 6 章写作建议

1. **专设小节**：「6.x OW 规则基线 vs 学习式分割」  
2. **表**：直接粘贴 `eddy_ow_rule_vs_yolo_compare.md` §1–§3  
3. **图**：`eddy_ow_rule_vs_yolo_compare.png`（RGB / 伪标签 / OW / YOLO 并列）  
4. **文字**：先承认 OW 复现上限，再写 YOLO 的三条工程价值（蒸馏、免疫超参、部署）  
5. **避免**：「YOLO 全面优于 OW」；「小涡旋召回显著提升」（与本实验相反）

---

## 5. 复现

```powershell
Set-Location F:\创赛
python scripts/eddy_compare_ow_rule_vs_yolo.py --splits val,test
# 无 GPU/权重时仅 OW：
python scripts/eddy_compare_ow_rule_vs_yolo.py --splits val --skip-yolo --max-samples 10
```

产物：

- `submission/tables/eddy_ow_rule_vs_yolo_compare.{md,json,csv}`
- `submission/figures/eddy_ow_rule_vs_yolo_compare.png`

---

## 6. 后续可选增强（若答辩仍被追问）

| 实验 | 目的 |
|------|------|
| conf 扫描 OW vs YOLO P–R | 说明 YOLO 可通过 conf 权衡过分割 |
| NC 加噪 / 缺 U/V 仅 RGB | 证明 YOLO 在 **输入退化** 时仍可跑（OW 必须 U/V） |
| 人工标注 20–50 帧 | 唯一可 claim「优于 OW」的 **独立真值** 路径 |
| META 产品 IoU 对齐 | 外部参考（需人工下载对齐） |
