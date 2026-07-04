#!/usr/bin/env python3
"""汇总 V6 Phase B0 val 指标：Fair vs Proposed@k 主表 + Leakage/Rule 旁证。"""

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
    ("Fair-B0", "主基线", "outputs/eddy_v6_b0_fair/metrics_summary_val.json", True),
    ("Proposed-B0-k3", "关键消融 offset=3", "outputs/eddy_v6_b0_proposed_k3/metrics_summary_val.json", True),
    ("Proposed-B0-k1", "对照 offset=1", "outputs/eddy_v6_b0_proposed_k1/metrics_summary_val.json", True),
    ("Proposed-B0-k5", "边界 offset=5", "outputs/eddy_v6_b0_proposed_k5/metrics_summary_val.json", False),
    ("Leakage-B0", "U/V 旁证", "outputs/eddy_v6_b0_leakage/metrics_summary_val.json", False),
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
    out: dict[str, float | None] = {k: None for k in _METRIC_KEYS}
    if not rule_json.is_file():
        return out
    with rule_json.open(encoding="utf-8") as f:
        data = json.load(f)
    row = (data.get("by_split_method") or {}).get(f"{split}_ow_rule")
    if not row:
        return out
    p, r = float(row.get("precision", 0)), float(row.get("recall", 0))
    out["mask_mean_precision"] = round(p, 4)
    out["mask_mean_recall"] = round(r, 4)
    out["mask_mean_f1"] = round(float(row.get("f1", 0)), 4)
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
    return "；".join(parts) if parts else "（待 eval 完成后计算）"


def main() -> None:
    ap = argparse.ArgumentParser(description="V6 Phase B0 Fair vs Proposed 主表")
    ap.add_argument("--out-md", default="submission/tables/eddy_v6_b0_fair_vs_proposed.md")
    ap.add_argument("--rule-json", default="submission/tables/eddy_v6_b0_rule_compare.json")
    args = ap.parse_args()

    rows: list[tuple[str, str, bool, dict[str, float | None]]] = []
    for name, role, rel_path, ranked in _GROUPS:
        rows.append((name, role, ranked, _read_metrics(resolve_path(rel_path))))

    rows.append(("Rule (OW P24)", "Teacher 上界", False, _read_rule_teacher(resolve_path(args.rule_json))))

    fair_m = next(m for n, _, _, m in rows if n == "Fair-B0")
    k3_m = next(m for n, _, _, m in rows if n == "Proposed-B0-k3")
    k1_m = next(m for n, _, _, m in rows if n == "Proposed-B0-k1")

    manifest_path = resolve_path("submission/tables/eddy_v6_b0_stem_manifest.json")
    train_n = val_n = "355/356"
    if manifest_path.is_file():
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
        tr = mf.get("splits", {}).get("train", {}).get("n_stems")
        va = mf.get("splits", {}).get("val", {}).get("n_stems")
        if tr is not None and va is not None:
            train_n = f"{tr}/{va}"

    out_md = resolve_path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# V6 Phase B0 — Fair vs Proposed offset 消融（val 2024，50 epoch）",
        "",
        "> **主比较**：Fair vs Proposed-B0-k3（关键消融）；k1/k5 为 offset 对照。",
        "> Leakage 与 Rule 为旁证，**不参与排名**。",
        "",
        f"训练：2018 全年 k_max=5 交集（manifest {train_n} stems）；标签 OW(P24)；归一化方案 A。",
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
            "## 主结论（Proposed-B0-k3 − Fair-B0）",
            "",
            _delta(k3_m, fair_m),
            "",
            "## offset 对照（Proposed-B0-k1 − Fair-B0）",
            "",
            _delta(k1_m, fair_m),
            "",
            "## 脚注",
            "",
            "- stem 清单见 `submission/tables/eddy_v6_b0_stem_manifest.json`。",
            "- val 仅用 k_max=5 裁剪，**不**叠加 Phase A 的 skip_boundary_days。",
            "- Rule：同 B0 stem 子集；`--v6-teacher` 重放 OW P24。",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
