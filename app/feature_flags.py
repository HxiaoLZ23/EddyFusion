"""演示系统功能开关（环境变量，默认保守）。"""

from __future__ import annotations

import os


def show_hydro_ui() -> bool:
    """是否展示水文推理 / 水文大屏等入口。

    新水文模型未达预期时默认 **关闭**。恢复展示：
    - Streamlit：启动前设置 ``EDDYFUSION_SHOW_HYDRO=1``（或 ``true`` / ``yes``）
    - Vite：``.env`` 中 ``VITE_SHOW_HYDRO=true`` 并重新构建

    详见 ``docs/实验与结果归档/水文_其他指标与能用标准归档.md`` §5。
    """
    v = os.environ.get("EDDYFUSION_SHOW_HYDRO", "").strip().lower()
    return v in ("1", "true", "yes", "on")
