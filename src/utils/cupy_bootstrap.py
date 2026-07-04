"""Windows 上尽量为 CuPy 配置 DLL 搜索路径（不保证可 JIT 编译，需完整 CUDA Toolkit 时见文档）。"""

from __future__ import annotations

import os
import site
import sys
from typing import Any


def _dll_dirs() -> list[str]:
    dirs: list[str] = []
    for root in site.getsitepackages():
        for sub in (
            "nvidia/cuda_nvrtc/bin",
            "nvidia/cuda_runtime/bin",
            "nvidia/cublas/bin",
        ):
            d = os.path.join(root, *sub.split("/"))
            if os.path.isdir(d):
                dirs.append(d)
    try:
        import torch

        tl = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(tl):
            dirs.append(tl)
    except ImportError:
        pass
    return dirs


def bootstrap_cupy_dll_paths() -> list[str]:
    """在 ``import cupy`` 之前调用；返回已注册的目录。"""
    if sys.platform != "win32":
        return []
    added: list[str] = []
    for d in _dll_dirs():
        try:
            os.add_dll_directory(d)
            added.append(d)
        except OSError:
            continue
    return added


def probe_cupy_runtime() -> dict[str, Any]:
    """
    探测 CuPy 是否可真正跑 kernel（不仅 import）。
    返回 ok / error / dll_dirs / fix_hint。
    """
    bootstrap_cupy_dll_paths()
    out: dict[str, Any] = {"ok": False, "dll_dirs": _dll_dirs()}
    try:
        import cupy as cp
    except ImportError as e:
        out["error"] = f"import: {e}"
        out["fix_hint"] = "pip install cupy-cuda11x"
        return out

    out["cupy_version"] = cp.__version__
    try:
        if not cp.cuda.is_available():
            out["error"] = "cupy.cuda.is_available() is False"
            out["fix_hint"] = _fix_hint()
            return out
        x = cp.ones((8, 8), dtype=cp.float64)
        cp.gradient(x)
        cp.cuda.Device(0).synchronize()
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        out["fix_hint"] = _fix_hint()
        return out


def _fix_hint() -> str:
    return (
        "Windows 上 CuPy 除 cupy-cuda11x 外通常还需 NVIDIA CUDA Toolkit 11.8，并设置\n"
        "  CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\n"
        "  PATH 含 %CUDA_PATH%\\bin\n"
        "或改用 benchmark 的 --backend torch（与已安装的 torch+cu118 共用运行时）。\n"
        "可选: pip install nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11"
    )
