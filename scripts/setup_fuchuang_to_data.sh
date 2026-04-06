#!/usr/bin/env bash
# 将命题方「服创数据集」放到可访问路径，并在仓库根创建符号链接「服创数据集」，
# 这样无需改 config/data.yaml 的 paths.raw_root（默认相对路径「服创数据集」）。
#
# 模式（重要）:
#   FU_MODE=copy  — 默认：rsync/cp 到 /data/服创数据集（需要目标盘有足够空间）
#   FU_MODE=link  — 不复制：仓库根「服创数据集」直接指向你已有的目录（零空间占用，推荐 AutoDL 数据在 autodl-tmp 时）
#
# 用法示例:
#   # 数据已在 /root/autodl-tmp/服创数据集，勿再拷到满盘的 /data：
#   FU_MODE=link FU_CHUANG_SRC=/root/autodl-tmp/服创数据集 bash scripts/setup_fuchuang_to_data.sh
#
#   # 仍要拷到数据盘（先 df -h 确认空间）:
#   FU_CHUANG_SRC=/path/to/服创数据集 bash scripts/setup_fuchuang_to_data.sh
#
#   FU_DATA_ROOT=/mnt/ssd  可改变 copy 模式下的目标根目录（默认 /data）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="${1:-${FU_CHUANG_SRC:-}}"
FU_MODE="${FU_MODE:-copy}"

if [[ -z "${SOURCE}" ]]; then
  echo "错误: 请指定命题方数据目录。"
  echo "  空间不足勿拷到 /data，请用零拷贝:"
  echo "    FU_MODE=link FU_CHUANG_SRC=/root/autodl-tmp/服创数据集 bash scripts/setup_fuchuang_to_data.sh"
  echo "  或: bash scripts/setup_fuchuang_to_data.sh /path/to/服创数据集"
  exit 1
fi

if [[ ! -d "${SOURCE}" ]]; then
  echo "错误: 源目录不存在: ${SOURCE}"
  echo "提示: Linux 路径区分大小写，项目目录须为 EddyFusion（Fusion 的 F 大写），不要写成 Eddyfusion。"
  echo "      若数据已在仓库内 ~/autodl-tmp/EddyFusion/服创数据集/，则无需运行本脚本。"
  exit 1
fi

SOURCE="$(cd "${SOURCE}" && pwd)"
LINK_NAME="${ROOT}/服创数据集"

if [[ "${FU_MODE}" == "link" ]]; then
  echo "模式: link（不复制，仅创建符号链接）"
  echo "源（物理数据）: ${SOURCE}"
  echo "仓库内链接: ${LINK_NAME} -> ${SOURCE}"
  if [[ -e "${LINK_NAME}" ]] && [[ ! -L "${LINK_NAME}" ]]; then
    echo "警告: ${LINK_NAME} 已存在且非符号链接，跳过。"
    exit 1
  fi
  ln -sfn "${SOURCE}" "${LINK_NAME}"
  echo "已创建: ${LINK_NAME} -> ${SOURCE}"
else
  DATA_ROOT="${FU_DATA_ROOT:-/data}"
  DEST="${DATA_ROOT%/}/服创数据集"

  echo "模式: copy（将同步到 ${DEST}，请确保目标盘空间充足）"
  echo "源: ${SOURCE}"
  echo "目标: ${DEST}"

  # 粗检：源占用 vs 目标可用空间（需 du/df，失败则仅提示）
  if command -v du >/dev/null 2>&1 && command -v df >/dev/null 2>&1; then
    if ! mkdir -p "${DATA_ROOT}" 2>/dev/null; then
      sudo mkdir -p "${DATA_ROOT}"
      sudo chown "$(id -u):$(id -g)" "${DATA_ROOT}" 2>/dev/null || true
    fi
    SRC_KB="$(du -sk "${SOURCE}" 2>/dev/null | awk '{print $1}')"
    mkdir -p "${DEST}" 2>/dev/null || sudo mkdir -p "${DEST}"
    AVAIL_KB="$(df -Pk "${DEST}" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "${SRC_KB}" && -n "${AVAIL_KB}" ]] && [[ "${AVAIL_KB}" -gt 0 ]]; then
      # 预留约 10% 余量（小文件与文件系统开销）
      NEED="$(( SRC_KB + SRC_KB / 10 ))"
      if [[ "${AVAIL_KB}" -lt "${NEED}" ]]; then
        echo ""
        echo "错误: 目标盘可用空间不足（约需 ${NEED} KB，可用约 ${AVAIL_KB} KB）。"
        echo "  rsync 会报 No space left on device。"
        echo "  若数据已在 ${SOURCE}，请勿再复制，改用零拷贝:"
        echo "    FU_MODE=link FU_CHUANG_SRC=${SOURCE} bash scripts/setup_fuchuang_to_data.sh"
        echo "  或清理 ${DATA_ROOT} / 扩容后再试。"
        exit 1
      fi
    fi
  fi

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

  if [[ -e "${LINK_NAME}" ]] && [[ ! -L "${LINK_NAME}" ]]; then
    echo "警告: ${LINK_NAME} 已存在且非符号链接，跳过创建链接。"
  else
    ln -sfn "${DEST}" "${LINK_NAME}"
    echo "已创建: ${LINK_NAME} -> ${DEST}"
  fi
fi

mkdir -p \
  "${ROOT}/data/processed/hydro" \
  "${ROOT}/data/processed/stats" \
  "${ROOT}/data/processed/eddy" \
  "${ROOT}/data/processed/anomaly"

echo ""
echo "下一步（在仓库根 ${ROOT} 下执行），生成水文 npz 示例:"
echo "  python -m src.preprocess.hydro_dataset --config config/hydro_hycom.yaml --from-nc --data-config config/data.yaml --max-daily-files 120 --stride 1"
echo ""
DEST_HINT="${DEST:-${SOURCE}}"
echo "若不用符号链接，可将 config/data.yaml 中 paths.raw_root 改为绝对路径:"
echo "  raw_root: \"${DEST_HINT}\""
