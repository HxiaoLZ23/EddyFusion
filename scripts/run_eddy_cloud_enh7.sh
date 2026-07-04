#!/usr/bin/env bash
# 云端：7 通道物理增强（无 Mask）导出 → 训练 → val/test 评估
# 在仓库根目录执行：bash scripts/run_eddy_cloud_enh7.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DATA="data/processed/eddy_enh7"
OUT_WT="AutoDL/outputs/eddy_enh7"

echo "== [1/4] 导出 YOLO-seg + 7ch npy（与 3ch 同伪标签口径） =="
python -m src.preprocess.eddy_yolo_export \
  --data-config config/data.yaml \
  --out "$OUT_DATA" \
  --stack-physics-npy \
  --physics-channels 7 \
  --time-stride 15

echo "== [2/4] 就绪检查 =="
python scripts/check_eddy_ready.py --dataset-yaml "$OUT_DATA/dataset.yaml"

echo "== [3/4] 训练 YOLOv8-seg（channels=7, pretrained=false） =="
python -u -m src.eddy.train --config config/eddy_enh7.yaml

echo "== [4/4] val / test 评估 =="
python -m src.eddy.eval \
  --config config/eddy_enh7.yaml \
  --ckpt "$OUT_WT/best.pt" \
  --splits val,test

echo "完成。指标：$OUT_WT/metrics_summary_val.json , metrics_summary_test.json"
echo "论文口径：勿将旧 8ch 0.826 与本次 7ch 混为同一主结论；0.826 仅可作模型-伪标签一致性上界说明。"
