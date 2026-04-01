"""规则模板预警报告。"""

from __future__ import annotations


def render_report(*args, **kwargs) -> str:
    raise NotImplementedError("anomaly.report")
