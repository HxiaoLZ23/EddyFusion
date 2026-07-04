"""24h 超前 eval 辅助函数测试。"""

from __future__ import annotations

import numpy as np

from src.anomaly.horizon_eval import horizon_steps_from_hours


def test_horizon_steps_24h_at_3h_step() -> None:
    assert horizon_steps_from_hours(24, 3) == 8


def test_horizon_steps_3h_at_3h_step() -> None:
    assert horizon_steps_from_hours(3, 3) == 1


def test_horizon_steps_1h_at_3h_step() -> None:
    assert horizon_steps_from_hours(1, 3) == 1


def test_align_wave_doubles_resolution() -> None:
    from src.anomaly.grid_eval import _align_wave_to_wind

    wave = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    out = _align_wave_to_wind(wave, 6, 6)
    assert out.shape == (4, 6, 6)
