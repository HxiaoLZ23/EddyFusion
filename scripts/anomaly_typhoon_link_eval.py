#!/usr/bin/env python3
"""
IBTrACS 测试年（默认 2024）台风「关联召回」轻量评测。

口径（与论文一致）：
- **中心任务**：风浪异常识别（LSTM 一步预测 + 残差/分级为运行链扩展）；
- **本脚本**：在**已知异常时空窗 = 真值台风 bbox/时段** 的理想条件下，
  评测 `link_anomaly_to_typhoon` 能否在 Top-K 中检索到对应 `event_id`。
- **不是**命题全文「端到端台风分类 Recall」：不含格点级 3σ 报警、不含 LSTM 漏报。

依赖：
  data/processed/anomaly/typhoon_kb/events.json
  （由 `scripts/build_typhoon_kb.py` 或 `scripts/seed_typhoon_kb_demo.py` 生成）

示例：
  python scripts/anomaly_typhoon_link_eval.py
  python scripts/anomaly_typhoon_link_eval.py --test-years 2024 --top-k 10 --min-peak-wind-kt 34
  python scripts/anomaly_typhoon_link_eval.py --out-json submission/tables/anomaly_typhoon_link_recall_2024.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.detect import link_anomaly_to_typhoon, run_detect
from src.utils.config import load_yaml, resolve_path


def _parse_time(text: str) -> datetime | None:
    raw = str(text).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _event_year(event: dict[str, Any]) -> int | None:
    season = event.get("season")
    if season is not None and str(season).strip().isdigit():
        return int(str(season).strip())
    st = _parse_time(str(event.get("start_time", "")))
    return st.year if st else None


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到台风事件索引: {path}\n"
            "请先执行其一：\n"
            "  python scripts/build_typhoon_kb.py --source-csv <IBTrACS.csv>\n"
            "  python scripts/seed_typhoon_kb_demo.py   # 演示用 DEMO 事件（含 2024）"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"events.json 应为数组: {path}")
    return data


def _filter_gt_events(
    events: list[dict[str, Any]],
    *,
    test_years: set[int],
    min_peak_wind_kt: float,
    basin_allow: set[str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        yr = _event_year(e)
        if yr is None or yr not in test_years:
            continue
        peak = float(e.get("peak_wind_kt") or 0.0)
        if peak < min_peak_wind_kt:
            continue
        if basin_allow is not None:
            b = str(e.get("basin", "")).strip().upper()
            if b and b not in basin_allow:
                continue
        out.append(e)
    return out


def _anomaly_result_from_gt_event(event: dict[str, Any], *, require_level: str | None = None) -> dict[str, Any]:
    """用真值台风时空窗构造 anomaly_result（Oracle 定位，测检索层上限）。"""
    ar: dict[str, Any] = {
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "lon_min": float(event.get("lon_min", 0)),
        "lon_max": float(event.get("lon_max", 0)),
        "lat_min": float(event.get("lat_min", 0)),
        "lat_max": float(event.get("lat_max", 0)),
        "wind_mean": 0.0,
        "wind_std": 1.0,
        "wave_mean": 0.0,
        "wave_std": 1.0,
        # 保证分级链有信号（high）
        "wind_residual": 6.0,
        "wave_residual": 6.0,
        "event_id": event.get("event_id"),
    }
    if require_level:
        ar["_require_level"] = require_level
    return ar


def _hit_in_candidates(gt_id: str, candidates: list[dict[str, Any]]) -> bool:
    gid = str(gt_id).strip()
    for c in candidates:
        if str(c.get("event_id", "")).strip() == gid:
            return True
    return False


def run_link_recall_eval(
    *,
    events_path: Path,
    test_years: set[int],
    top_k: int,
    min_peak_wind_kt: float,
    basin_allow: set[str] | None,
    use_full_detect: bool,
) -> dict[str, Any]:
    events = _load_events(events_path)
    gt_list = _filter_gt_events(
        events,
        test_years=test_years,
        min_peak_wind_kt=min_peak_wind_kt,
        basin_allow=basin_allow,
    )

    hits = 0
    rows: list[dict[str, Any]] = []
    for e in gt_list:
        eid = str(e.get("event_id", ""))
        ar = _anomaly_result_from_gt_event(e)
        if use_full_detect:
            out = run_detect(anomaly_result=ar, auto_link_typhoon=True, events_json_path=str(events_path), top_k=top_k)
            candidates = out.get("typhoon_link", {}).get("candidates") or []
            level = out.get("anomaly_result", {}).get("anomaly_level")
        else:
            link = link_anomaly_to_typhoon(anomaly_result=ar, events_json_path=str(events_path), top_k=top_k)
            candidates = link.get("candidates") or []
            level = None

        ok = _hit_in_candidates(eid, candidates)
        hits += int(ok)
        rows.append(
            {
                "event_id": eid,
                "name": e.get("name", ""),
                "season": e.get("season", ""),
                "peak_wind_kt": e.get("peak_wind_kt"),
                "hit_top_k": ok,
                "top1_id": (candidates[0].get("event_id") if candidates else None),
                "n_candidates": len(candidates),
                "anomaly_level": level,
            }
        )

    n = len(gt_list)
    recall = float(hits / n) if n else 0.0
    return {
        "metric_name": "typhoon_link_recall_oracle_bbox",
        "description": (
            "在真值台风时空窗作为异常查询框时，Top-K 检索是否包含该 event_id；"
            "评测台风弱关联解释层，非端到端台风检测。"
        ),
        "events_json": str(events_path),
        "test_years": sorted(test_years),
        "top_k": top_k,
        "min_peak_wind_kt": min_peak_wind_kt,
        "basin_filter": sorted(basin_allow) if basin_allow else None,
        "n_gt_events": n,
        "n_hit": hits,
        "link_recall": round(recall, 6),
        "use_full_detect": use_full_detect,
        "per_event": rows,
    }


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 模块 C：台风事件关联召回（Oracle 时空窗 + KB 检索）",
        "",
        "> **不是**命题端到端「台风识别准确率」；**不是** `src.anomaly.eval` 的 MAE/RMSE。",
        "> 在已知真值台风 bbox/时段 下测试 `link_anomaly_to_typhoon` 的 Top-K 命中率。",
        "",
        f"- 事件索引：`{summary['events_json']}`",
        f"- 测试年：{summary['test_years']}",
        f"- Top-K：{summary['top_k']}",
        f"- 最低峰值风速（kt）：{summary['min_peak_wind_kt']}",
        f"- 真值事件数：**{summary['n_gt_events']}**",
        f"- 命中数：**{summary['n_hit']}**",
        f"- **关联 Recall = {summary['link_recall']}**",
        "",
        "| event_id | name | season | peak_wind_kt | hit@K | top1_id |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary.get("per_event", []):
        hit = "Y" if r.get("hit_top_k") else "N"
        lines.append(
            f"| {r.get('event_id','')} | {r.get('name','')} | {r.get('season','')} | "
            f"{r.get('peak_wind_kt','')} | {hit} | {r.get('top1_id','')} |"
        )
    lines.append("")
    lines.append("复现：`python scripts/anomaly_typhoon_link_eval.py`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="台风 KB 关联 Recall（测试年，Oracle 查询框）")
    ap.add_argument("--events-json", type=str, default="")
    ap.add_argument("--test-years", type=str, default="", help="逗号分隔，默认读 config/data.yaml anomaly_year_split.test_years")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-peak-wind-kt", type=float, default=34.0, help="过滤弱系统，默认热带风暴档")
    ap.add_argument("--basin", type=str, default="", help="可选，逗号分隔，如 WP,NI")
    ap.add_argument("--full-detect", action="store_true", help="走 run_detect（含 3σ 分级）再联动")
    ap.add_argument("--out-json", type=str, default="submission/tables/anomaly_typhoon_link_recall_2024.json")
    ap.add_argument("--out-md", type=str, default="submission/tables/anomaly_typhoon_link_recall_2024.md")
    args = ap.parse_args()

    demo_cfg = load_yaml("app/config/demo.yaml")
    ty_cfg = demo_cfg.get("typhoon_link", {}) if isinstance(demo_cfg.get("typhoon_link"), dict) else {}
    events_json = args.events_json.strip() or str(ty_cfg.get("events_json_path", "data/processed/anomaly/typhoon_kb/events.json"))
    events_path = resolve_path(events_json)

    if args.test_years.strip():
        test_years = {int(x.strip()) for x in args.test_years.split(",") if x.strip()}
    else:
        data_cfg = load_yaml("config/data.yaml")
        ys = (data_cfg.get("anomaly_year_split") or {}).get("test_years") or [2024]
        test_years = {int(x) for x in ys}

    basin_allow: set[str] | None = None
    if args.basin.strip():
        basin_allow = {x.strip().upper() for x in args.basin.split(",") if x.strip()}

    top_k = int(args.top_k) if args.top_k > 0 else int(ty_cfg.get("default_top_k", 10))

    summary = run_link_recall_eval(
        events_path=events_path,
        test_years=test_years,
        top_k=top_k,
        min_peak_wind_kt=float(args.min_peak_wind_kt),
        basin_allow=basin_allow,
        use_full_detect=bool(args.full_detect),
    )

    out_json = resolve_path(args.out_json)
    out_md = resolve_path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(out_md, summary)

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(
        f"link_recall={summary['link_recall']} "
        f"({summary['n_hit']}/{summary['n_gt_events']} events, years={summary['test_years']})"
    )


if __name__ == "__main__":
    main()
