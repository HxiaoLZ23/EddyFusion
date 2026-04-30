#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/check_eddy_ready.py --dataset-yaml data/processed/eddy/dataset.yaml
python -m src.eddy.train --config config/eddy.yaml
