#!/usr/bin/env bash
# 将命题方「服创数据集」同步到服务器 /data，并在仓库根创建符号链接 服创数据集 → /data/服创数据集
# 这样无需改 config/data.yaml 的 paths.raw_root（默认相对路径「服创数据集」）。
#
# 用法:
#   export FU_CHUANG_SRC=/root/autodl-tmp/服创数据集   # 或你上传/解压后的目录
#   bash scripts/setup_fuchuang_to_data.sh
#
# 或显式传入源路径:
#   bash scripts/setup_fuchuang_to_data.sh /path/to/服创数据集
#
# 若不能使用 /data（无权限），可设环境变量:
#   FU_DATA_ROOT=/mnt/disk1  则目标为 $FU_DATA_ROOT/服创数据集

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="${1:-${FU_CHUANG_SRC:-}}"
if [[ -z "${SOURCE}" ]]; then
  echo "错误: 请指定命题方数据目录。"
  echo "  方式一: FU_CHUANG_SRC=/path/to/服创数据集 bash scripts/setup_fuchuang_to_data.sh"
  echo "  方式二: bash scripts/setup_fuchuang_to_data.sh /path/to/服创数据集"
  exit 1
fi

if [[ ! -d "${SOURCE}" ]]; then
  echo "错误: 源目录不存在: ${SOURCE}"
  exit 1
fi

DATA_ROOT="${FU_DATA_ROOT:-/data}"
DEST="${DATA_ROOT%/}/服创数据集"
LINK_NAME="${ROOT}/服创数据集"

echo "源: ${SOURCE}"
echo "目标（物理目录）: ${DEST}"
echo "仓库内链接: ${LINK_NAME} -> ${DEST}"

if ! mkdir -p "${DATA_ROOT}" 2>/dev/null; then
  echo "创建 ${DATA_ROOT} 需要提升权限…"
  sudo mkdir -p "${DATA_ROOT}"
  sudo chown "$(id -u):$(id -g)" "${DATA_ROOT}" 2>/dev/null || true
fi

if ! mkdir -p "${DEST}" 2>/dev/null; then
  sudo mkdir -p "${DEST}"
  sudo chown "$(id -u):$(id -g)" "${DEST}" 2>/dev/null || true
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "${SOURCE%/}/" "${DEST}/"
else
  echo "未找到 rsync，使用 cp -a"
  cp -a "${SOURCE%/}/." "${DEST}/"
fi

# 仓库根「服创数据集」→ 实际数据目录（覆盖旧链接）
if [[ -e "${LINK_NAME}" ]] && [[ ! -L "${LINK_NAME}" ]]; then
  echo "警告: ${LINK_NAME} 已存在且非符号链接，跳过创建链接，请自行合并或改名。"
else
  ln -sfn "${DEST}" "${LINK_NAME}"
  echo "已创建: ${LINK_NAME} -> ${DEST}"
fi

# 预处理输出目录（空仓库或未生成时）
mkdir -p \
  "${ROOT}/data/processed/hydro" \
  "${ROOT}/data/processed/stats" \
  "${ROOT}/data/processed/eddy" \
  "${ROOT}/data/processed/anomaly"

echo ""
echo "下一步（在仓库根 ${ROOT} 下执行），生成水文 npz 示例:"
echo "  python -m src.preprocess.hydro_dataset --config config/hydro_hycom.yaml --from-nc --data-config config/data.yaml --max-daily-files 120 --stride 1"
echo ""
echo "若希望不用符号链接、直接在配置里写绝对路径，可将 config/data.yaml 中 paths.raw_root 改为:"
echo "  raw_root: \"${DEST}\""
