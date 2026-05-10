#!/usr/bin/env python3
"""
从多次 `python -m src.eddy.eval --splits val,test` 产出的 JSON 汇总 mask mAP50 表，
写入 submission/tables/ 供材料与论文对照。

示例：
  python scripts/eddy_write_val_test_map_table.py \\
    --row baseline:outputs/eddy/metrics_summary_val.json:outputs/eddy/metrics_summary_test.json \\
    --row enh8_mc:outputs/eddy_enh/metrics_summary_val.json:outputs/eddy_enh/metrics_summary_test.json \\
    --out-csv submission/tables/eddy_map_val_test.csv \\
    --out-md submission/tables/eddy_map_val_test.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_path


def _aggregate_ablation_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """按 mode 汇总 n_det 均值、toy 灵敏度（不含 mAP）。"""
    by_mode: dict[str, list[float]] = {}
    tta_frac: dict[str, list[float]] = {}
    with csv_path.open(encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            mode = row.get("mode") or ""
            try:
                n = float(row.get("n_det") or 0)
            except ValueError:
                n = 0.0
            by_mode.setdefault(mode, []).append(n)
            tta = str(row.get("tta_any", "")).lower() in ("true", "1", "yes")
            tta_frac.setdefault(mode, []).append(1.0 if tta else 0.0)
    out: dict[str, dict[str, float]] = {}
    for m, lst in by_mode.items():
        nmean = sum(lst) / max(len(lst), 1)
        tlist = tta_frac.get(m, [])
        tmean = sum(tlist) / max(len(tlist), 1) if tlist else 0.0
        out[m] = {"mean_n_det": round(nmean, 4), "tta_any_frac": round(tmean, 4)}
    return out


def _read_map50(json_path: Path) -> float | None:
    if not json_path.is_file():
        return None
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("metrics") or {}
    v = m.get("mask_map50")
    return float(v) if v is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description="聚合 eddy eval JSON 为 val/test mAP50 表")
    ap.add_argument(
        "--row",
        dest="rows",
        action="append",
        default=[],
        metavar="NAME:VAL_JSON[:TEST_JSON]",
        help="可重复；TEST 省略则该实验无 test 列",
    )
    ap.add_argument("--out-csv", type=str, default="submission/tables/eddy_map_val_test.csv")
    ap.add_argument("--out-md", type=str, default="submission/tables/eddy_map_val_test.md")
    ap.add_argument(
        "--ablation-csv",
        type=str,
        default="",
        help="可选：eddy_inference_ablate.py 的 CSV；在 md 末尾附加「推理侧灵敏度」均值表（非 mAP）",
    )
    args = ap.parse_args()

    parsed: list[tuple[str, Path, Path | None]] = []
    for raw in args.rows:
        parts = raw.split(":")
        if len(parts) < 2:
            raise SystemExit(f"--row 格式错误: {raw}")
        name = parts[0].strip()
        val_p = resolve_path(parts[1].strip())
        test_p = resolve_path(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else None
        parsed.append((name, val_p, test_p))

    if not parsed:
        raise SystemExit("请至少提供一行 --row name:val.json[:test.json]")

    out_csv = resolve_path(args.out_csv)
    out_md = resolve_path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, str | float | None]] = []
    for name, val_p, test_p in parsed:
        v50 = _read_map50(val_p)
        t50 = _read_map50(test_p) if test_p is not None else None
        rows_out.append(
            {
                "experiment": name,
                "val_mask_map50": v50 if v50 is None else round(v50, 6),
                "test_mask_map50": None if t50 is None else round(t50, 6),
                "val_json": str(val_p),
                "test_json": "" if test_p is None else str(test_p),
            }
        )

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fn = ["experiment", "val_mask_map50", "test_mask_map50", "val_json", "test_json"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows_out)

    lines = [
        "# 涡旋分割 mAP@0.5（Ultralytics seg，固定划分）",
        "",
        "| 实验 | val mask mAP50 | test mask mAP50 |",
        "| --- | --- | --- |",
    ]
    for r in rows_out:
        vv = "" if r["val_mask_map50"] is None else str(r["val_mask_map50"])
        tt = "" if r["test_mask_map50"] is None else str(r["test_mask_map50"])
        lines.append(f"| {r['experiment']} | {vv} | {tt} |")
    lines.append("")
    lines.append("说明：推理侧频域/unsharp、多尺度 TTA 的对照若未烘焙进验证集，`scripts/eddy_inference_ablate.py` 仅反映检测计数/置信度灵敏度；正式口径以本表 `eval` 为准。")

    if args.ablation_csv:
        ac = resolve_path(args.ablation_csv)
        if ac.is_file():
            agg = _aggregate_ablation_csv(ac)
            lines.extend(["", "## 附录：推理消融（均值检测数 / TTA 命中占比，非 mAP）", "", "| mode | mean n_det | tta_any_frac |", "| --- | --- | --- |"])
            for m in sorted(agg.keys()):
                v = agg[m]
                lines.append(f"| {m} | {v['mean_n_det']} | {v['tta_any_frac']} |")
        else:
            lines.extend(["", f"<!-- 未找到 ablation CSV: {ac} -->"])

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
