"""DTW 异常窗与 match_mode 配置（app/config/demo.yaml → typhoon_link）。"""

from __future__ import annotations

from typing import Any

from src.anomaly.eddy_typhoon_bridge import safe_float

DEFAULT_DTW_MATCH_MODE = "regional_mean_obs_vs_ibtracs_center"
LEGACY_DTW_MATCH_MODE = "wind_residual_vs_ibtracs_track"

_DEFAULTS: dict[str, Any] = {
    "dtw_match_mode": DEFAULT_DTW_MATCH_MODE,
    "dtw_window_tau": 1.5,
    "dtw_window_min_len": 2,
    "dtw_window_gap_merge": 1,
    "dtw_window_pad": 4,
    "dtw_window_fallback_peak_half_width": 5,
}


def load_dtw_link_config() -> dict[str, Any]:
    """读取 typhoon_link 下 DTW 相关键；缺失项用默认值。"""
    out = dict(_DEFAULTS)
    try:
        from src.anomaly.eddy_typhoon_bridge import _load_typhoon_link_yaml_cfgs

        _, demo_cfg = _load_typhoon_link_yaml_cfgs()
        ty_cfg = demo_cfg.get("typhoon_link") if isinstance(demo_cfg.get("typhoon_link"), dict) else {}
        if isinstance(ty_cfg, dict):
            if ty_cfg.get("dtw_match_mode"):
                out["dtw_match_mode"] = str(ty_cfg["dtw_match_mode"]).strip()
            out["dtw_window_tau"] = safe_float(ty_cfg.get("dtw_window_tau"), float(out["dtw_window_tau"]))
            out["dtw_window_min_len"] = max(1, int(safe_float(ty_cfg.get("dtw_window_min_len"), float(out["dtw_window_min_len"]))))
            out["dtw_window_gap_merge"] = max(0, int(safe_float(ty_cfg.get("dtw_window_gap_merge"), float(out["dtw_window_gap_merge"]))))
            out["dtw_window_pad"] = max(0, int(safe_float(ty_cfg.get("dtw_window_pad"), float(out["dtw_window_pad"]))))
            out["dtw_window_fallback_peak_half_width"] = max(
                1,
                int(safe_float(ty_cfg.get("dtw_window_fallback_peak_half_width"), float(out["dtw_window_fallback_peak_half_width"]))),
            )
    except Exception:
        pass
    return out
