#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m src.preprocess.netcdf_io --config config/data.yaml
python -m src.preprocess.eddy_dataset --config config/data.yaml
python -m src.preprocess.hydro_dataset --config config/data.yaml
python -m src.preprocess.anomaly_dataset --config config/data.yaml
