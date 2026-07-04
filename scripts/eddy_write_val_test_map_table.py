#!/usr/bin/env python3
"""
从多次 `python -m src.eddy.eval --splits val,test` 产出的 JSON 汇总 mask 指标表，
写入 submission/tables/ 供材料与论文对照。

除 ``mask_map50`` 外，若 JSON 中存在则一并写出：
``mask_map50_95``、``mask_map75``、``mask_mean_precision``、``mask_mean_recall``、``mask_mean_f1``。

示例：
  python scripts/eddy_write_val_test_map_table.py \\
    --row baseline:outputs/eddy/metrics_summary_val.json:outputs/eddy/metrics_summary_test.json \\
    --row enh8_mc:AutoDL/outputs/eddy_enh/metrics_summary_val.json:AutoDL/outputs/eddy_enh/metrics_summary_test.json \\
    --out-csv submission/tables/eddy_map_val_test.csv \\
    --out-md submission/tables/eddy_map_val_test.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import resolve_path

# 与 src.eddy.eval 写出的 ``metrics`` 子对象键一致（顺序即 CSV 列顺序）
_EXTRA_FLOAT_KEYS: tuple[str, ...] = (
    "mask_map50",
    "mask_map50_95",
    "mask_map75",
    "mask_mean_precision",
    "mask_mean_recall",
    "mask_mean_f1",
)


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


def _read_eddy_metrics(json_path: Path) -> dict[str, Any | None]:
    if not json_path.is_file():
        return {k: None for k in _EXTRA_FLOAT_KEYS}
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("metrics") or {}
    out: dict[str, Any | None] = {}
    for k in _EXTRA_FLOAT_KEYS:
        v = m.get(k)
        if v is None:
            out[k] = None
        else:
            try:
                out[k] = round(float(v), 6)
            except (TypeError, ValueError):
                out[k] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="聚合 eddy eval JSON 为 val/test 分割指标表")
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

    csv_cols: list[str] = ["experiment"]
    for k in _EXTRA_FLOAT_KEYS:
        csv_cols.append(f"val_{k}")
        csv_cols.append(f"test_{k}")
    csv_cols.extend(["val_json", "test_json"])

    rows_out: list[dict[str, str | float | None]] = []
    for name, val_p, test_p in parsed:
        vm = _read_eddy_metrics(val_p)
        tm = _read_eddy_metrics(test_p) if test_p is not None else {k: None for k in _EXTRA_FLOAT_KEYS}
        row: dict[str, str | float | None] = {"experiment": name}
        for k in _EXTRA_FLOAT_KEYS:
            row[f"val_{k}"] = vm.get(k)
            row[f"test_{k}"] = tm.get(k) if test_p is not None else None
        row["val_json"] = str(val_p)
        row["test_json"] = "" if test_p is None else str(test_p)
        rows_out.append(row)

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        w.writerows(rows_out)

    def _cell(v: Any) -> str:
        return "" if v is None else str(v)

    lines = [
        "# 涡旋分割验证指标（Ultralytics YOLO-seg，固定划分）",
        "",
        "列名前缀：`val_` / `test_`；指标键与 `metrics_summary_*.json` 内 `metrics` 一致。",
        "",
        "| 实验 | split | mAP50 | mAP50-95 | mAP75 | mean P | mean R | mean F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    metric_keys = [k for k in _EXTRA_FLOAT_KEYS if k != "mask_map50"]
    for r in rows_out:
        for split_label, prefix in (("val", "val_"), ("test", "test_")):
            if split_label == "test" and not r.get("test_json"):
                continue
            cells = [_cell(r["experiment"]), split_label, _cell(r[f"{prefix}mask_map50"])]
            for mk in metric_keys:
                cells.append(_cell(r[f"{prefix}{mk}"]))
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "说明：``mask_mean_precision`` / ``mask_mean_recall`` / ``mask_mean_f1`` 为 Ultralytics 对各类别指标的均值；"
        "推理侧频域/unsharp、多尺度 TTA 若未烘焙进验证集，`scripts/eddy_inference_ablate.py` 仅反映检测计数/置信度灵敏度；"
        "正式分割评测口径以本表 `eval` 为准。"
    )

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
