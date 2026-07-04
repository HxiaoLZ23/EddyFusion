"""DashScope 密钥：环境变量优先，其次 ``config/dashscope.local.json``（勿提交 Git）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.utils.config import project_root

_LOCAL_JSON = project_root() / "config" / "dashscope.local.json"
_APPLIED = False


def dashscope_local_json_path() -> Path:
    return _LOCAL_JSON


def load_dashscope_file() -> dict[str, Any]:
    p = _LOCAL_JSON
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def apply_dashscope_env_from_file(*, force: bool = False) -> bool:
    """
    将本地 JSON 中的 DASHSCOPE_* 写入 os.environ（仅当对应环境变量未设置时）。
    返回是否读到有效文件。
    """
    global _APPLIED
    if _APPLIED and not force:
        return _LOCAL_JSON.is_file()
    _APPLIED = True
    data = load_dashscope_file()
    if not data:
        return False
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_MODEL", "DASHSCOPE_WORKSPACE"):
        if key in data and str(data[key]).strip() and not os.environ.get(key, "").strip():
            os.environ[key] = str(data[key]).strip()
    return True


def get_dashscope_api_key(explicit: str | None = None) -> str:
    apply_dashscope_env_from_file()
    return (explicit or os.environ.get("DASHSCOPE_API_KEY", "") or "").strip()


def get_dashscope_model(explicit: str | None = None) -> str:
    apply_dashscope_env_from_file()
    return (explicit or os.environ.get("DASHSCOPE_MODEL", "") or "").strip()
