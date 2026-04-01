from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.config import ensure_dir, resolve_path


def write_metrics_json(
    path: str | Path,
    *,
    module: str,
    level: int,
    metrics: dict[str, Any],
    passed: bool | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    """各模块 eval 汇总 JSON，字段需与《A09-项目开发文档》5.3 一致。"""
    p = resolve_path(path)
    ensure_dir(p.parent)
    payload: dict[str, Any] = {
        "module": module,
        "level": int(level),
        "metrics": metrics,
        "passed": passed,
        "tags": tags or {},
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
