#!/usr/bin/env python3
"""汇总 3ch / 7ch / 各 ablation profile 的 val/test mask_map50。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.config import resolve_path

ROWS = [
    ("3ch (baseline)", "outputs/eddy_cloud_fair"),
    ("+zeta (4_bgr_zeta)", "outputs/eddy_ablation/4_bgr_zeta"),
    ("+ow (4_bgr_ow)", "outputs/eddy_ablation/4_bgr_ow"),
    ("+grad (5_bgr_grad)", "outputs/eddy_ablation/5_bgr_grad"),
    ("+zeta+ow (5_no_grad)", "outputs/eddy_ablation/5_no_grad"),
    ("+zeta+grad (6_no_ow)", "outputs/eddy_ablation/6_no_ow"),
    ("+ow+grad (6_no_zeta)", "outputs/eddy_ablation/6_no_zeta"),
    ("+all (7ch)", "outputs/eddy_enh7_cloud_fair"),
]


def _read_map50(p: Path) -> float | None:
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    m = data.get("metrics") or {}
    v = m.get("mask_map50")
    return round(float(v), 6) if v is not None else None


def main() -> None:
    lines = [
        "# 3ch 增通道消融 mAP 汇总（相对 3ch 基线）",
        "",
        "| 实验 | val mask mAP@0.5 | test mask mAP@0.5 | 备注 |",
        "| --- | --- | --- | --- |",
    ]
    for name, out_rel in ROWS:
        base = resolve_path(out_rel)
        val = _read_map50(base / "metrics_summary_val.json")
        test = _read_map50(base / "metrics_summary_test.json")
        note = "缺失" if val is None and test is None else ""
        lines.append(
            f"| {name} | {val if val is not None else '-'} | {test if test is not None else '-'} | {note} |"
        )

    out = REPO / "submission/tables/eddy_ablation_7ch_matrix.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
