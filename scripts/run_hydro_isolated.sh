#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

LEVEL="${1:-l2}"

case "$LEVEL" in
  l2)
    CFG="config/hydro_hycom_l2.yaml"
    CKPT="outputs/hydro_l2/best.pt"
    ;;
  l1)
    CFG="config/hydro_hycom_l1.yaml"
    CKPT="outputs/hydro_l1/best.pt"
    ;;
  l0)
    CFG="config/hydro_hycom_l0.yaml"
    CKPT="outputs/hydro_l0/best.pt"
    ;;
  *)
    echo "用法: bash scripts/run_hydro_isolated.sh [l2|l1|l0]"
    exit 1
    ;;
esac

echo "[hydro isolated] level=${LEVEL}, config=${CFG}"
if [[ "$LEVEL" != "l2" ]]; then
  echo "[WARN] 当前 MoT/AttnRes/EOS 仍在接入中，L1/L0 主要用于隔离实验链路。"
fi

python -m src.hydro.train --config "$CFG"
CKPT_TO_USE="$CKPT"
if [ ! -f "$CKPT_TO_USE" ]; then
  LAST_CKPT="${CKPT%best.pt}last.pt"
  if [ -f "$LAST_CKPT" ]; then
    echo "[WARN] best.pt 不存在，回退使用 last.pt: ${LAST_CKPT}"
    CKPT_TO_USE="$LAST_CKPT"
  else
    echo "[ERROR] 未找到可用权重: ${CKPT} / ${LAST_CKPT}"
    exit 1
  fi
fi

python -m src.hydro.eval --config "$CFG" --ckpt "$CKPT_TO_USE" --split val
python -m src.hydro.eval --config "$CFG" --ckpt "$CKPT_TO_USE" --split test
python scripts/export_material_table.py

echo "[hydro isolated] done: ${LEVEL}"
