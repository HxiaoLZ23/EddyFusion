from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def _project_root() -> Path:
    # app/services/*.py -> app/services -> app -> repo root
    return Path(__file__).resolve().parents[2]


@dataclass
class MetricsItem:
    exists: bool
    raw: dict[str, Any]
    message: str


class MetricsService:
    def __init__(self) -> None:
        root = _project_root()
        self.targets: dict[str, Path] = {
            "hydro_val": root / "outputs" / "hydro" / "metrics_summary_val.json",
            "hydro_test": root / "outputs" / "hydro" / "metrics_summary_test.json",
            "anomaly": root / "outputs" / "anomaly" / "metrics_summary.json",
        }

    def _read_one(self, path: Path) -> MetricsItem:
        if not path.is_file():
            return MetricsItem(False, {}, f"missing: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return MetricsItem(True, raw if isinstance(raw, dict) else {"value": raw}, "ok")
        except Exception as e:
            return MetricsItem(False, {}, f"read_error: {path} -> {e}")

    def load_all(self) -> dict[str, MetricsItem]:
        return {name: self._read_one(path) for name, path in self.targets.items()}

