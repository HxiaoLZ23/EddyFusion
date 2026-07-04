#!/usr/bin/env python3
"""汇总 V6 Phase A val 指标：Fair vs Proposed 主表 + Leakage/Rule 旁证。

用法：
  python scripts/eddy_write_v6_fair_vs_proposed.py
  python scripts/eddy_write_v6_fair_vs_proposed.py --out-md submission/tables/eddy_v6_fair_vs_proposed.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_path

_METRIC_KEYS = (
    "mask_map50",
    "mask_map50_95",
    "mask_mean_precision",
    "mask_mean_recall",
    "mask_mean_f1",
)

_GROUPS: tuple[tuple[str, str, str, bool], ...] = (
    ("Fair", "新基线", "outputs/eddy_v6_fair/metrics_summary_val.json", True),
    ("Proposed", "新方案（三时刻 ADT）", "outputs/eddy_v6_proposed/metrics_summary_val.json", True),
    ("Leakage", "U/V 旁证", "outputs/eddy_v6_leakage/metrics_summary_val.json", False),
)


def _read_metrics(json_path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in _METRIC_KEYS}
    if not json_path.is_file():
        return out
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("metrics") or {}
    for k in _METRIC_KEYS:
        v = m.get(k)
        if v is not None:
            try:
                out[k] = round(float(v), 4)
            except (TypeError, ValueError):
                out[k] = None
    return out


def _read_rule_teacher(rule_json: Path, split: str = "val") -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "mask_map50": None,
        "mask_map50_95": None,
        "mask_mean_precision": None,
        "mask_mean_recall": None,
        "mask_mean_f1": None,
    }
    if not rule_json.is_file():
        return out
    with rule_json.open(encoding="utf-8") as f:
        data = json.load(f)
    key = f"{split}_ow_rule"
    row = (data.get("by_split_method") or {}).get(key)
    if not row:
        return out
    p, r = float(row.get("precision", 0)), float(row.get("recall", 0))
    f1 = float(row.get("f1", 0))
    out["mask_mean_precision"] = round(p, 4)
    out["mask_mean_recall"] = round(r, 4)
    out["mask_mean_f1"] = round(f1, 4)
    # Rule 无 YOLO mAP；用 matched mIoU 作参考列（脚注说明）
    miou = row.get("mean_iou_matched")
    if miou is not None:
        out["mask_map50"] = round(float(miou), 4)
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _delta(proposed: dict[str, float | None], fair: dict[str, float | None]) -> str:
    parts: list[str] = []
    for k, label in (
        ("mask_map50", "ΔmAP50"),
        ("mask_mean_precision", "ΔP"),
        ("mask_mean_recall", "ΔR"),
    ):
        pv, fv = proposed.get(k), fair.get(k)
        if pv is not None and fv is not None:
            parts.append(f"{label}={pv - fv:+.3f}")
    return "；".join(parts) if parts else "（待 Proposed eval 完成后计算）"


def main() -> None:
    ap = argparse.ArgumentParser(description="V6 Phase A Fair vs Proposed 主表")
    ap.add_argument("--out-md", default="submission/tables/eddy_v6_fair_vs_proposed.md")
    ap.add_argument(
        "--rule-json",
        default="submission/tables/eddy_v6_rule_compare.json",
        help="Rule Teacher 旁证 JSON",
    )
    args = ap.parse_args()

    rows: list[tuple[str, str, bool, dict[str, float | None]]] = []
    for name, role, rel_path, ranked in _GROUPS:
        m = _read_metrics(resolve_path(rel_path))
        rows.append((name, role, ranked, m))

    rule_m = _read_rule_teacher(resolve_path(args.rule_json))
    rows.append(("Rule (OW P24)", "Teacher 上界", False, rule_m))

    fair_m = next(m for n, _, _, m in rows if n == "Fair")
    prop_m = next(m for n, _, _, m in rows if n == "Proposed")

    out_md = resolve_path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# V6 Phase A — Fair vs Proposed（val 2024，50 epoch 烟测）",
        "",
        "> **主比较**：Fair（单帧 ADT×3）vs Proposed（三时刻 ADT）。",
        "> Leakage（ADT+U/V）与 Rule（OW P24 重放）为旁证，**不参与排名**。",
        "",
        "训练：2018Q1（90 帧）；评测：2024 val（364 帧）；标签 OW(P24)；归一化方案 A。",
        "",
        "| 组别 | 角色 | 参与排名 | mAP50 | mAP50-95 | P | R | F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, role, ranked, m in rows:
        rank_cell = "✓ 主比" if ranked else "✗"
        lines.append(
            f"| {name} | {role} | {rank_cell} | {_fmt(m['mask_map50'])} | "
            f"{_fmt(m['mask_map50_95'])} | {_fmt(m['mask_mean_precision'])} | "
            f"{_fmt(m['mask_mean_recall'])} | {_fmt(m['mask_mean_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## 主结论（Proposed − Fair）",
            "",
            _delta(prop_m, fair_m),
            "",
            "## 脚注",
            "",
            "- **Rule**：相对伪标签 instance IoU≥0.5 重放；P/R≈1 为 Teacher 上界；mAP 列用 matched mIoU 示意，非 YOLO mAP。",
            "- **Leakage**：含 U/V 伪彩输入，仅作信息泄漏旁证，不得与 Fair/Proposed 直接比优劣。",
            "- 指标来自 `python -m src.eddy.eval --splits val` 的 `metrics_summary_val.json`。",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
