#!/usr/bin/env python3
"""统计 DTW 异常事件窗 fallback 率与窗长（Goal: dtw_window_stats.json）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.detect import build_wind_dtw_query_curve, compute_series_anomaly_segments  # noqa: E402
from src.anomaly.dtw_config import load_dtw_link_config  # noqa: E402
from src.anomaly.windwave_nc_bridge import extract_wind_wave_companion_from_netcdf  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402


def _stats_for_nc(nc_path: Path) -> dict[str, Any] | None:
    companion = extract_wind_wave_companion_from_netcdf(nc_path)
    if companion is None:
        return None
    wo = companion.get("demo_wind_observed")
    wp = companion.get("demo_wind_predicted")
    ho = companion.get("demo_wave_observed")
    hp = companion.get("demo_wave_predicted")
    if not all(isinstance(x, list) for x in (wo, wp, ho, hp)):
        return None
    cfg = load_dtw_link_config()
    segments = compute_series_anomaly_segments(
        wind_observed=[float(v) for v in wo],
        wind_predicted=[float(v) for v in wp],
        wave_observed=[float(v) for v in ho],
        wave_predicted=[float(v) for v in hp],
    )
    curve, meta = build_wind_dtw_query_curve(
        wind_observed=[float(v) for v in wo],
        wind_predicted=[float(v) for v in wp],
        wave_observed=[float(v) for v in ho],
        wave_predicted=[float(v) for v in hp],
        segments=segments,
        mode=str(cfg.get("dtw_match_mode")),
        dtw_config=cfg,
    )
    window = meta.get("window") if isinstance(meta.get("window"), dict) else {}
    n = len(wo)
    win_len = len(curve)
    return {
        "nc_path": str(nc_path),
        "n_steps": n,
        "window_len": win_len,
        "window_ratio": round(win_len / n, 4) if n else 0.0,
        "fallback_reason": meta.get("fallback_reason"),
        "tau": window.get("tau"),
        "t_start_padded": window.get("t_start_padded"),
        "t_end_padded": window.get("t_end_padded"),
    }


def main() -> None:
    candidates = [
        resolve_path("data/demo/offline_test_merged_t300.nc"),
        resolve_path("data/demo/windwave_demo.nc"),
    ]
    rows: list[dict[str, Any]] = []
    for p in candidates:
        if p.is_file():
            row = _stats_for_nc(p)
            if row:
                rows.append(row)

    n = len(rows)
    n_fallback = sum(1 for r in rows if r.get("fallback_reason"))
    med_len = sorted(r["window_len"] for r in rows)[n // 2] if n else 0
    summary = {
        "n_nc_sampled": n,
        "fallback_rate": round(n_fallback / n, 4) if n else None,
        "median_window_len": med_len,
        "dtw_config": load_dtw_link_config(),
        "samples": rows,
    }
    out = resolve_path("submission/tables/dtw_window_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
