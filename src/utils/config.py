from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """仓库根目录（含 config/、src/）。

    与运行时的「当前目录」无关；云端常为 ``/root/autodl-tmp/EddyFusion``，本地因人而异。
    配置与数据中凡非绝对路径均相对本目录解析，勿在业务代码中写死某环境的绝对前缀。
    """
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def resolve_path(path: str | Path) -> Path:
    """相对路径一律相对 ``project_root()``；绝对路径原样返回（用于用户显式指定）。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pick_device(preferred: str) -> str:
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
