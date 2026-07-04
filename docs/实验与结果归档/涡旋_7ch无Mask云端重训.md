# 涡旋 7 通道物理增强（无 Mask）— 云端重训说明

## 1. 答辩口径（回应 Mask shortcut）

- **导师质疑成立的情形**：若把 **OW/连通域二值 Mask** 或 **与伪标签同源的候选 Mask** 作为 YOLO **输入通道**，模型可能学会「复制标签」而非从 ADT/流场学习，不宜作为「物理融合有效」的主证据。
- **本仓库当前 8ch 栈（`build_physics_stacked_hw8`）不含 Mask 输入**，通道为：BGR(ADT,U,V) + ζ + Lap(ADT) + OW + 粗尺度残差 + |∇ADT|。Mask 仅用于 **伪标签生成**（`eddy_yolo_export` 中 OW 投票 → 轮廓），不写入 npy。
- **仍建议主实验改为 7ch**：按答辩意见，去掉 Lap/残差/梯度幅等易与「工程消融」混淆的通道，改为论文表述一致的 **GradX、GradY**，并单独目录 `eddy_enh7` 重训，与历史 `eddy_enh`（8ch）结果隔离。

## 2. 7 通道定义（`build_physics_stacked_hw7`）

| 下标 | 内容 |
|------|------|
| 0–2 | BGR：V, U, ADT（与 3ch RGB 同色标） |
| 3 | 相对涡度 ζ |
| 4 | Okubo–Weiss |
| 5 | ∂ADT/∂x（分位归一化） |
| 6 | ∂ADT/∂y（分位归一化） |

**不包含**：二值 Mask、伪标签多边形、Laplacian、多尺度残差（除非另做消融）。

## 3. 云端一条龙命令

```bash
cd /root/autodl-tmp/EddyFusion   # 按云机实际仓库根调整
bash scripts/run_eddy_cloud_enh7.sh
```

分步：

```bash
# 导出（与 3ch 同划分、同 OW 伪标签参数；仅 npy 为 7 通道）
python -m src.preprocess.eddy_yolo_export \
  --data-config config/data.yaml \
  --out data/processed/eddy_enh7 \
  --stack-physics-npy --physics-channels 7 \
  --time-stride 15

python -u -m src.eddy.train --config config/eddy_enh7.yaml

python -m src.eddy.eval \
  --config config/eddy_enh7.yaml \
  --ckpt AutoDL/outputs/eddy_enh7/best.pt \
  --splits val,test
```

产物（相对仓库根）：

- 数据：`data/processed/eddy_enh7/`
- 权重与指标：`AutoDL/outputs/eddy_enh7/best.pt`、`metrics_summary_val.json`、`metrics_summary_test.json`

## 4. 论文 / 表 5-4 表述建议

| 结果 | 建议表述 |
|------|----------|
| 8ch val≈0.839 / test≈0.826（`eddy_enh`） | **模型–伪标签一致性**（物理通道增强 + 与 OW 伪标签同分布特征）；**不作为**「已严格排除 Mask 泄漏」的最终识别精度 |
| 3ch 基线 0.762 / 0.730 | 可与 7ch 并列为主对比（同划分、同伪标签） |
| 7ch（`eddy_enh7` 重训后） | **主结论**：3ch vs 7ch 物理通道，mask mAP@0.5 |

图 4-2 对比脚本（7ch 训练完成后）：

```bash
python scripts/eddy_plot_3ch_vs_8ch_input_compare.py \
  --baseline-dir AutoDL/outputs/eddy \
  --enh-dir AutoDL/outputs/eddy_enh7 \
  --conf 0.5
```

（脚本文件名仍为 `8ch`，`--enh-dir` 指向 7ch 目录即可。）

## 5. 与 3ch 对比的公平性

- 同一 `train/val/test` 时间划分（`config/data.yaml` + nc 文件名规则）
- 同一 OW 投票伪标签流程
- 仅差：输入通道（3 vs 7）与权重（`eddy` vs `eddy_enh7`）

## 6. 需人工完成

- 在 **AutoDL GPU** 上跑满 `run_eddy_cloud_enh7.sh`（本地通常无全量 NC）
- 将 `metrics_summary_*.json`、`best.pt`、可选 `eval_val/val_batch0_pred.jpg` 同步回本地 `AutoDL/outputs/eddy_enh7/`
- 更新论文表 5-4、图 4-2 与 `毕业设计材料/定稿1.md` 中「8 通道」主表述为「7 通道物理增强（无 Mask 输入）」
