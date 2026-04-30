#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 第二轮建议：扩大样本覆盖，仍保留多分位投票过滤
python -m src.preprocess.eddy_dataset --export-yolo --data-config config/data.yaml \
  --time-stride 20 \
  --max-frames-per-file 20 \
  --vote-percentiles 10,15,20,25,30 \
  --vote-min 2 \
  --min-area-px 60 \
  --max-area-frac 0.18 \
  --approx-eps-frac 0.003 \
  --max-instances 50

python scripts/check_eddy_ready.py --dataset-yaml data/processed/eddy/dataset.yaml
python -m src.eddy.train --config config/eddy.yaml
python -m src.eddy.eval --config config/eddy.yaml --ckpt outputs/eddy/best.pt
python scripts/export_material_table.py

echo "Round2 complete: eddy train/eval/table refreshed."
