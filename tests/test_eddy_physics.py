from __future__ import annotations

import numpy as np

from src.eddy.stacked_physics import (
    ablation_profile_channels,
    build_physics_stacked_ablation,
    build_physics_stacked_hw6_no_ow,
    build_physics_stacked_hw7,
    build_physics_stacked_hw8,
    relative_vorticity_and_okubo_weiss_from_uv,
)


def _synthetic_fields(shape: tuple[int, int] = (32, 32)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[-1.0:1.0:complex(shape[0]), -1.0:1.0:complex(shape[1])]
    adt = np.exp(-(x**2 + y**2) * 3.0).astype(np.float32)
    u = (-y + 0.1 * x).astype(np.float32)
    v = (x + 0.1 * y).astype(np.float32)
    return adt, u, v


def test_relative_vorticity_and_okubo_weiss_are_finite() -> None:
    _adt, u, v = _synthetic_fields()

    zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u, v)

    assert zeta.shape == u.shape
    assert ow.shape == u.shape
    assert np.isfinite(zeta).all()
    assert np.isfinite(ow).all()
    assert float(np.abs(zeta).mean()) > 0


def test_build_physics_stacked_hw7_shape_dtype_and_range() -> None:
    adt, u, v = _synthetic_fields()
    zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u, v)

    stacked = build_physics_stacked_hw7(adt, u, v, zeta, ow)

    assert stacked.shape == (32, 32, 7)
    assert stacked.dtype == np.float32
    assert np.isfinite(stacked).all()
    assert float(stacked.min()) >= 0.0
    assert float(stacked.max()) <= 1.0


def test_build_physics_stacked_hw6_no_ow_is_six_channels() -> None:
    adt, u, v = _synthetic_fields()
    zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u, v)
    stacked = build_physics_stacked_hw6_no_ow(adt, u, v, zeta)
    assert stacked.shape == (32, 32, 6)


def test_ablation_profiles_shapes() -> None:
    adt, u, v = _synthetic_fields()
    zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u, v)
    for profile, expected_c in (
        ("4_bgr_zeta", 4),
        ("4_bgr_ow", 4),
        ("5_bgr_grad", 5),
        ("5_no_grad", 5),
        ("6_no_ow", 6),
        ("6_no_zeta", 6),
    ):
        stacked = build_physics_stacked_ablation(profile, adt, u, v, zeta, ow)
        assert stacked.shape == (32, 32, expected_c)
        assert ablation_profile_channels(profile) == expected_c


def test_build_physics_stacked_hw8_shape_dtype_and_range() -> None:
    adt, u, v = _synthetic_fields()
    zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u, v)

    stacked = build_physics_stacked_hw8(adt, u, v, zeta, ow)

    assert stacked.shape == (32, 32, 8)
    assert stacked.dtype == np.float32
    assert np.isfinite(stacked).all()
    assert float(stacked.min()) >= 0.0
    assert float(stacked.max()) <= 1.0
