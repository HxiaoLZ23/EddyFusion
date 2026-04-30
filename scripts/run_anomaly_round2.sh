#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 命题方年份划分：train 2014-2023 / test 2024 / val 2025（由 config/data.yaml 控制）
python -m src.preprocess.anomaly_dataset \
  --config config/anomaly.yaml \
  --data-config config/data.yaml \
  --from-nc --year-split --stride 1

python -m src.anomaly.train --config config/anomaly.yaml
python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split val
python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split test
python scripts/export_material_table.py

echo "Anomaly round2 complete: preprocess/train/eval/table refreshed."
