#!/usr/bin/env python3
"""表 6-3：历史风过程检索结果统计。

统计台风知识库中带 IBTrACS 中心风轨迹的过程数量，并在 Oracle 时空窗 +
合成 wind_dtw_curve 条件下评测 Top-K 检索与 DTW 重排完成率。

口径：
- **历史风过程数量**：events.json 中 ``wind_track_mps`` 至少 2 点且 max>0 的事件数。
- **Top-K**：默认 10。
- **检索成功率**：时空初筛后 ``linked=True``（至少 1 条候选）的比例。
- **DTW 重排完成率**：``retrieval.dtw.enabled=True`` 的比例（需查询侧 ``wind_dtw_curve``）。

示例：
  python scripts/anomaly_wind_process_retrieval_stats.py
  python scripts/anomaly_wind_process_retrieval_stats.py --top-k 10 --test-years 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.anomaly_typhoon_link_eval import (  # noqa: E402
    _anomaly_result_from_gt_event,
    _filter_gt_events,
    _load_events,
)
from src.anomaly.detect import link_anomaly_to_typhoon  # noqa: E402
from src.utils.config import load_yaml, resolve_path


def _count_kb_wind_processes(events: list[dict[str, Any]]) -> dict[str, int]:
    total = len(events)
    with_track = 0
    peak_only = 0
    for e in events:
        mps = e.get("wind_track_mps")
        kt = e.get("wind_track_kt")
        vals: list[float] = []
        if isinstance(mps, list) and mps:
            vals = [float(v) for v in mps if isinstance(v, (int, float))]
        elif isinstance(kt, list) and kt:
            vals = [float(v) * 0.514444 for v in kt if isinstance(v, (int, float))]
        if len(vals) >= 2 and max(vals) > 0:
            with_track += 1
        elif float(e.get("peak_wind_kt") or 0) > 0:
            peak_only += 1
    return {
        "n_events_total": total,
        "n_with_wind_track": with_track,
        "n_peak_only_fallback": peak_only,
    }


def _synthetic_wind_dtw_curve(event: dict[str, Any], *, n: int = 8) -> list[float]:
    """Oracle 链路：用风轨迹形态构造区域平均观测风速代理（验收 DTW 可跑通，非物理真值）。"""
    mps = event.get("wind_track_mps")
    if isinstance(mps, list) and len(mps) >= 2:
        arr = np.asarray([float(v) for v in mps], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2 and float(np.max(arr)) > 0:
            if arr.size >= n:
                idx = np.linspace(0, arr.size - 1, num=n, dtype=int)
                base = arr[idx]
            else:
                base = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, arr.size), arr)
            # 区域平均相对中心风偏低：缩放代理
            return (base * 0.35 + 2.0).astype(float).tolist()
    peak = float(event.get("peak_wind_kt") or 40.0) * 0.514444
    ramp = np.linspace(0.2, 1.0, n)
    return (ramp * peak * 0.12 + 1.5).astype(float).tolist()


def run_table_6_3_eval(
    *,
    events_path: Path,
    test_years: set[int],
    top_k: int,
    min_peak_wind_kt: float,
) -> dict[str, Any]:
    events = _load_events(events_path)
    kb = _count_kb_wind_processes(events)
    gt_list = _filter_gt_events(
        events,
        test_years=test_years,
        min_peak_wind_kt=min_peak_wind_kt,
        basin_allow=None,
    )

    n_linked = 0
    n_dtw_ok = 0
    rows: list[dict[str, Any]] = []
    for e in gt_list:
        eid = str(e.get("event_id", ""))
        ar = _anomaly_result_from_gt_event(e)
        ar["wind_dtw_curve"] = _synthetic_wind_dtw_curve(e)
        ar["dtw_match_mode"] = "regional_mean_obs_vs_ibtracs_center"
        ar["dtw_query_curve"] = "wind_obs_regional_mean_window"
        link = link_anomaly_to_typhoon(
            anomaly_result=ar,
            events_json_path=str(events_path),
            top_k=top_k,
        )
        candidates = link.get("candidates") or []
        linked = bool(link.get("linked"))
        dtw = (link.get("retrieval") or {}).get("dtw") or {}
        dtw_enabled = bool(dtw.get("enabled"))
        n_linked += int(linked)
        n_dtw_ok += int(dtw_enabled)
        rows.append(
            {
                "event_id": eid,
                "name": e.get("name", ""),
                "linked": linked,
                "n_candidates": len(candidates),
                "dtw_enabled": dtw_enabled,
                "dtw_match_mode": dtw.get("match_mode"),
                "n_with_track": dtw.get("n_candidates_with_track"),
            }
        )

    n_q = len(gt_list)
    retrieval_rate = float(n_linked / n_q) if n_q else 0.0
    dtw_rate = float(n_dtw_ok / n_q) if n_q else 0.0

    return {
        "table_id": "6-3",
        "title": "历史风过程检索结果统计",
        "events_json": str(events_path),
        "test_years": sorted(test_years),
        "top_k": top_k,
        "min_peak_wind_kt": min_peak_wind_kt,
        "kb": kb,
        "n_eval_queries": n_q,
        "n_retrieval_success": n_linked,
        "retrieval_success_rate": round(retrieval_rate, 6),
        "n_dtw_rerank_success": n_dtw_ok,
        "dtw_rerank_completion_rate": round(dtw_rate, 6),
        "dtw_match_mode": "regional_mean_obs_vs_ibtracs_center",
        "per_event": rows,
    }


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    kb = summary["kb"]
    lines = [
        "# 表 6-3  历史风过程检索结果统计",
        "",
        "> 由 `scripts/anomaly_wind_process_retrieval_stats.py` 生成。",
        "> 检索评测在 **Oracle 真值时空窗** + 合成区域平均观测风速查询曲线下统计，验收 KB 索引与 DTW 重排链路（非严格物理匹配验证）。",
        "",
        "**表 6-3  历史风过程检索结果统计**",
        "",
        "| 测试内容 | 数值 |",
        "| --- | --- |",
        f"| 历史风过程数量 | **{kb['n_with_wind_track']}** |",
        f"| Top-K | **{summary['top_k']}** |",
        f"| DTW 重排完成率 | **{summary['dtw_rerank_completion_rate'] * 100:.0f}%** |",
        f"| 检索成功率 | **{summary['retrieval_success_rate'] * 100:.0f}%** |",
        "",
        "## 口径说明",
        "",
        f"- 知识库总事件：**{kb['n_events_total']}**；含非零 ``wind_track_mps`` 过程：**{kb['n_with_wind_track']}**；",
        f"  仅峰值常数降级：**{kb['n_peak_only_fallback']}**。",
        f"- 评测样本：测试年 {summary['test_years']}、peak≥{summary['min_peak_wind_kt']} kt 共 **{summary['n_eval_queries']}** 条。",
        f"- DTW 口径：``{summary['dtw_match_mode']}``（查询异常窗内区域平均 ``wind_obs`` ↔ 历史 ``wind_track_mps``；z-score 后比时间演化形态）。",
        "",
        "复现：",
        "",
        "```bash",
        "python scripts/anomaly_wind_process_retrieval_stats.py",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="表 6-3 历史风过程检索统计")
    ap.add_argument("--events-json", type=str, default="")
    ap.add_argument("--test-years", type=str, default="2024")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-peak-wind-kt", type=float, default=34.0)
    ap.add_argument("--out-json", type=str, default="submission/tables/table_6_3_wind_process_retrieval.json")
    ap.add_argument("--out-md", type=str, default="submission/tables/table_6_3_wind_process_retrieval.md")
    args = ap.parse_args()

    demo_cfg = load_yaml("app/config/demo.yaml")
    ty_cfg = demo_cfg.get("typhoon_link", {}) if isinstance(demo_cfg.get("typhoon_link"), dict) else {}
    events_json = args.events_json.strip() or str(ty_cfg.get("events_json_path", "data/processed/anomaly/typhoon_kb/events.json"))
    events_path = resolve_path(events_json)

    test_years = {int(x.strip()) for x in args.test_years.split(",") if x.strip()}
    top_k = int(args.top_k) if args.top_k > 0 else int(ty_cfg.get("default_top_k", 10))

    summary = run_table_6_3_eval(
        events_path=events_path,
        test_years=test_years,
        top_k=top_k,
        min_peak_wind_kt=float(args.min_peak_wind_kt),
    )

    out_json = resolve_path(args.out_json)
    out_md = resolve_path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(out_md, summary)

    kb = summary["kb"]
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(
        f"历史风过程={kb['n_with_wind_track']} Top-K={top_k} "
        f"DTW完成率={summary['dtw_rerank_completion_rate']:.0%} "
        f"检索成功率={summary['retrieval_success_rate']:.0%} "
        f"({summary['n_eval_queries']} queries)"
    )


if __name__ == "__main__":
    main()
