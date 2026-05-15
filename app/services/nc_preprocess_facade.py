"""统一预处理 Facade：NC 接入后的任务路由与摘要（深度预处理逐步接入 src/preprocess）。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from services.nc_ingest_service import summarize_nc_file


class NcTaskBranch(str, Enum):
    EDDY = "eddy"
    HYDRO = "hydro"
    WINDWAVE = "windwave"
    FULL_CHAIN = "full_chain"


def describe_for_branch(paths: list[str | Path], branch: NcTaskBranch) -> dict[str, Any]:
    """
    对给定 NC 路径做摘要，并标注目标分支（尚未调用重型 preprocess，仅占位）。
    多文件时按列表顺序逐一摘要。
    """
    items: list[dict[str, Any]] = []
    for p in paths:
        items.append(summarize_nc_file(p))
    return {
        "branch": branch.value,
        "n_files": len(items),
        "files": items,
        "note": "完整格点预处理（重采样、滑窗、Z-score）将按分支调用 src/preprocess/*；当前仅元数据摘要。",
    }


def route_preprocess_stub(paths: list[str | Path], branch: NcTaskBranch) -> dict[str, Any]:
    """占位：后续替换为真实 preprocess 管道输出（npz 路径、张量形状等）。"""
    return describe_for_branch(paths, branch)
