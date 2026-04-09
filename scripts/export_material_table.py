from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _flatten_metrics(prefix: str, obj: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.extend(_flatten_metrics(p, v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.extend(_flatten_metrics(p, v))
    else:
        out.append((prefix, str(obj)))
    return out


def _collect_metric_files(root: Path) -> list[Path]:
    cands = [
        root / "hydro" / "metrics_summary_val.json",
        root / "hydro" / "metrics_summary_test.json",
        root / "hydro" / "metrics_summary.json",
        root / "anomaly" / "metrics_summary_val.json",
        root / "anomaly" / "metrics_summary_test.json",
        root / "anomaly" / "metrics_summary.json",
        root / "eddy" / "metrics_summary_val.json",
        root / "eddy" / "metrics_summary_test.json",
        root / "eddy" / "metrics_summary.json",
    ]
    return [p for p in cands if p.is_file()]


def _row_from_payload(path: Path, payload: dict[str, Any]) -> list[dict[str, str]]:
    module = str(payload.get("module", path.parent.name))
    level = str(payload.get("level", "-"))
    passed = str(payload.get("passed", "-"))
    tags = payload.get("tags", {})
    split = "-"
    if isinstance(tags, dict) and "eval_split" in tags:
        split = str(tags.get("eval_split"))
    elif "val" in path.stem:
        split = "val"
    elif "test" in path.stem:
        split = "test"

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {"value": metrics}
    flat = _flatten_metrics("", metrics)
    rows: list[dict[str, str]] = []
    for k, v in flat:
        rows.append(
            {
                "module": module,
                "split": split,
                "level": level,
                "metric": k,
                "value": v,
                "passed": passed,
                "source_file": str(path),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fields = ["module", "split", "level", "metric", "value", "passed", "source_file"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    lines = [
        "# 系统运行效果指标表",
        "",
        "| 模块 | split | level | 指标 | 数值 | 是否达标 | 来源文件 |",
        "|------|-------|-------|------|------|----------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['module']} | {r['split']} | {r['level']} | {r['metric']} | {r['value']} | {r['passed']} | `{r['source_file']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出企业材料可贴用指标表（CSV+Markdown）")
    parser.add_argument("--metrics-root", type=str, default="outputs", help="默认读取 outputs/*/metrics_summary*.json")
    parser.add_argument(
        "--extra-root",
        type=str,
        default="AutoDL/outputs",
        help="可选第二指标目录（若存在则一并读取）",
    )
    parser.add_argument("--out-dir", type=str, default="submission/tables", help="导出目录")
    parser.add_argument("--name", type=str, default="system_metrics_table", help="导出文件名前缀")
    args = parser.parse_args()

    root = project_root()
    roots = [resolve_path(args.metrics_root)]
    extra = resolve_path(args.extra_root)
    if extra.is_dir():
        roots.append(extra)

    files: list[Path] = []
    for r in roots:
        files.extend(_collect_metric_files(r))
    # 去重并按路径排序，保证输出稳定
    files = sorted(set(files), key=lambda p: str(p))
    if not files:
        raise FileNotFoundError(
            f"未找到指标文件。检查 {args.metrics_root} / {args.extra_root} 下是否有 metrics_summary*.json"
        )

    rows: list[dict[str, str]] = []
    for fp in files:
        payload = _read_json(fp)
        if payload is None:
            continue
        rows.extend(_row_from_payload(fp, payload))

    out_dir = resolve_path(args.out_dir)
    csv_path = out_dir / f"{args.name}.csv"
    md_path = out_dir / f"{args.name}.md"
    _write_csv(csv_path, rows)
    _write_md(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"rows={len(rows)}, files={len(files)}")


if __name__ == "__main__":
    main()

