from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.anomaly.inference import (
    predict_wind_wave_from_series,
    rolling_predict,
    smooth_baseline,
    window_steps_from_cfg,
)
from src.anomaly.model import build_model
from src.utils.config import load_yaml


def test_smooth_baseline_preserves_endpoints() -> None:
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0], dtype=np.float64)
    y = smooth_baseline(x)
    assert y[0] == x[0]
    assert y[-1] == x[-1]


def test_predict_fallback_when_checkpoint_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    feat = np.array([[1.0, 0.5], [2.0, 0.6], [1.5, 0.55]], dtype=np.float32)
    out = predict_wind_wave_from_series(feat, ckpt_path=missing)
    assert out.prediction_backend == "smooth_fallback"
    assert out.fallback_reason == "checkpoint_missing"
    assert len(out.wind_observed) == 3
    assert len(out.wind_predicted) == 3


def test_predict_fallback_when_series_too_short(tmp_path: Path) -> None:
    cfg = load_yaml("config/anomaly.yaml")
    model = build_model(cfg)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)

    w, h, _ = window_steps_from_cfg(cfg, load_yaml("config/data.yaml"))
    need = w + h
    feat = np.random.randn(need - 1, 2).astype(np.float32)
    out = predict_wind_wave_from_series(feat, ckpt_path=ckpt, device="cpu")
    assert out.prediction_backend == "smooth_fallback"
    assert out.fallback_reason is not None
    assert "series_too_short" in out.fallback_reason


def test_rolling_predict_and_lstm_backend(tmp_path: Path) -> None:
    cfg = load_yaml("config/anomaly.yaml")
    model = build_model(cfg)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)

    w, h, _ = window_steps_from_cfg(cfg, load_yaml("config/data.yaml"))
    need = w + h
    t_len = need + 5
    rng = np.random.default_rng(42)
    feat = rng.standard_normal((t_len, 2)).astype(np.float32) * 0.1 + np.array([5.0, 1.2])

    out = predict_wind_wave_from_series(feat, ckpt_path=ckpt, device="cpu")
    assert out.prediction_backend == "lstm"
    assert out.fallback_reason is None
    assert len(out.wind_observed) == t_len - need + 1
    assert len(out.wind_predicted) == len(out.wind_observed)
    assert len(out.wave_observed) == len(out.wind_observed)

    dev = torch.device("cpu")
    model.eval()
    _, obs, pred, resid = rolling_predict(
        model,
        feat,
        window_steps=w,
        horizon_steps=h,
        device=dev,
    )
    assert obs.shape == pred.shape == resid.shape
    assert obs.shape[0] == t_len - need + 1


def test_windwave_nc_bridge_uses_prediction_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from netCDF4 import Dataset

    from src.anomaly import windwave_nc_bridge

    nc_path = tmp_path / "ww.nc"
    nt, ny, nx = 24, 4, 4
    with Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("lat", ny)
        ds.createDimension("lon", nx)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 2024-08-01 00:00:00"
        t[:] = np.arange(nt, dtype=np.float64) * 3.0
        u = ds.createVariable("u10", "f4", ("time", "lat", "lon"))
        v = ds.createVariable("v10", "f4", ("time", "lat", "lon"))
        swh = ds.createVariable("swh", "f4", ("time", "lat", "lon"))
        u[:] = 8.0
        v[:] = 1.0
        swh[:] = 1.5

    cfg = load_yaml("config/anomaly.yaml")
    model = build_model(cfg)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)

    monkeypatch.setattr(
        windwave_nc_bridge,
        "predict_wind_wave_from_series",
        lambda feat, **kw: predict_wind_wave_from_series(feat, ckpt_path=ckpt, device="cpu"),
    )
    companion = windwave_nc_bridge.extract_wind_wave_companion_from_netcdf(nc_path)
    assert companion is not None
    assert companion["prediction_backend"] == "lstm"
    assert len(companion["demo_wind_observed"]) >= 2
