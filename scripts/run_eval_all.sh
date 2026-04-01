#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m src.eddy.eval --config config/eddy.yaml --ckpt outputs/eddy/best.pt
python -m src.hydro.eval --config config/hydro.yaml --ckpt outputs/hydro/best.pt
python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt
