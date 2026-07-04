#!/usr/bin/env python3
"""
从 `python -m src.anomaly.eval --split val|test` 产出的 `metrics_summary_{val,test}.json`
聚合成对外统一表，写入 submission/tables/。

示例（默认路径与 config/anomaly.yaml 的 eval.metrics_file 约定一致）：
  python scripts/anomaly_export_metrics_table.py
  python scripts/anomaly_export_metrics_table.py \\
    --val-json outputs/anomaly/metrics_summary_val.json \\
    --test-json outputs/anomaly/metrics_summary_test.json \\
    --out-md submission/tables/anomaly_metrics_val_test.md \\
    --out-csv submission/tables/anomaly_metrics_val_test.csv
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


def _load_metrics(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("metrics") or {}
    keys = (
        "mae_wind",
        "mae_wave",
        "mae_avg",
        "rmse_wind",
        "rmse_wave",
        "rmse_avg",
    )
    out: dict[str, float] = {}
    for k in keys:
        if k in m and m[k] is not None:
            out[k] = float(m[k])
    return out if out else None


def _fmt(v: float | None, nd: int = 6) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="聚合 anomaly eval JSON 为 val/test 指标表")
    ap.add_argument(
        "--val-json",
        type=str,
        default="outputs/anomaly/metrics_summary_val.json",
    )
    ap.add_argument(
        "--test-json",
        type=str,
        default="outputs/anomaly/metrics_summary_test.json",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default="submission/tables/anomaly_metrics_val_test.md",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="submission/tables/anomaly_metrics_val_test.csv",
    )
    ap.add_argument(
        "--ckpt-note",
        type=str,
        default="outputs/anomaly/best.pt",
        help="写入 md 表脚注释的权重路径（材料口径）",
    )
    args = ap.parse_args()

    val_p = resolve_path(args.val_json)
    test_p = resolve_path(args.test_json)

    def _repo_rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return p.as_posix()
    vm = _load_metrics(val_p)
    tm = _load_metrics(test_p)

    rows = [
        ("MAE 风速", "mae_wind", vm, tm),
        ("MAE 波高", "mae_wave", vm, tm),
        ("MAE 平均", "mae_avg", vm, tm),
        ("RMSE 风速", "rmse_wind", vm, tm),
        ("RMSE 波高", "rmse_wave", vm, tm),
        ("RMSE 平均", "rmse_avg", vm, tm),
    ]

    out_md = resolve_path(args.out_md)
    out_csv = resolve_path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "key", "val", "test"])
        for label, key, vdict, tdict in rows:
            vv = vdict.get(key) if vdict else None
            tv = tdict.get(key) if tdict else None
            w.writerow([label, key, _fmt(vv), _fmt(tv)])

    lines = [
        "# 模块 C（风浪异常）回归指标汇总（val / test）",
        "",
        "| 指标 | val | test |",
        "|------|-----|------|",
    ]
    for label, key, vdict, tdict in rows:
        vv = vdict.get(key) if vdict else None
        tv = tdict.get(key) if tdict else None
        lines.append(f"| {label} | {_fmt(vv)} | {_fmt(tv)} |")

    lines.extend(
        [
            "",
            "## 复现与口径",
            "",
            f"- **权重（材料口径）**：`{args.ckpt_note}`（新键名 `wind_head` / `wave_head`）",
            "- **评估命令**：",
            "  - `python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split val`",
            "  - `python -m src.anomaly.eval --config config/anomaly.yaml --ckpt outputs/anomaly/best.pt --split test`",
            f"- **JSON 来源**：`{_repo_rel(val_p)}`、`{_repo_rel(test_p)}`（由 `eval` 自动写出）",
            "- **旧格式 checkpoint**：仅作加载兼容自检时 val MAE 可能极大，见 `docs/开发规划/后续开发工作清单_未完成项与云端L0专项.md` §1.3.1。",
            "",
        ]
    )
    if vm is None or tm is None:
        lines.append(
            "> **说明**：若上表为 `—`，请先完成训练并运行上述两条 `eval`，再执行本脚本覆盖表与 CSV。"
        )
    elif vm and tm:
        v_avg = vm.get("mae_avg")
        t_avg = tm.get("mae_avg")
        if (
            v_avg is not None
            and t_avg is not None
            and v_avg > 2.0
            and t_avg < 1.0
        ):
            lines.append(
                "> **提示**：当前 JSON 中 **val** 与 **test** 的 `mae_avg` 差距较大；请确认两个 JSON 均由同一 `best.pt`（新键名双头）生成。若为兼容自检用的旧格式权重，见工作清单 §1.3.1。"
            )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
