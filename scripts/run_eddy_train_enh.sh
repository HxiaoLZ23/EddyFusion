#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# 导出需带 --stack-physics-npy，见 config/eddy_enh.yaml paths.dataset_yaml
python scripts/check_eddy_ready.py --dataset-yaml data/processed/eddy_enh/dataset.yaml
python -m src.eddy.train --config config/eddy_enh.yaml
