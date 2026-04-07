from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.netcdf_io import open_netcdf_dataset
from src.utils.config import load_yaml, project_root, resolve_path


def generate_synthetic_anomaly(config_path: str | Path) -> None:
    cfg = load_yaml(config_path)
    rng = np.random.default_rng(int(cfg["meta"]["seed"]))
    paths = cfg["paths"]
    win = int(cfg["data"]["window_hours"])
    n_train, n_val, n_test = 400, 80, 80

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        # 特征：风速、波高；目标为下一时刻
        x = rng.standard_normal((n, win, 2)).astype(np.float32)
        # 累积随机游走近似时序
        x = np.cumsum(x, axis=1)
        y = rng.standard_normal((n, 2)).astype(np.float32) * 0.1 + x[:, -1] * 0.9
        return x, y

    key_map = {
        "train": paths["train_sequences"],
        "val": paths["val_sequences"],
        "test": paths["test_sequences"],
    }
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        xp = resolve_path(key_map[split])
        x, y = make(n)
        xp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(xp, X=x, y=y)
        print(f"wrote {xp} X={x.shape} y={y.shape}")

    idx = {
        "source": "synthetic",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    tip = resolve_path(paths["typhoon_index"])
    tip.parent.mkdir(parents=True, exist_ok=True)
    with tip.open("w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print(f"wrote {tip}")


def _calendar_year_from_path(p: Path) -> int | None:
    for part in p.parts:
        if part.isdigit() and len(part) == 4:
            y = int(part)
            if 1900 <= y <= 2100:
                return y
    return None


def discover_anomaly_nc_paths(
    raw_root: Path,
    subdir: str,
    max_daily_files: int | None = None,
    *,
    years: set[int] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[Path]:
    root = raw_root / subdir
    if not root.is_dir():
        raise FileNotFoundError(f"风浪数据目录不存在: {root}")
    files = sorted(root.rglob("*.nc"))
    files = [f for f in files if "__MACOSX" not in f.parts]
    if years is not None:
        files = [f for f in files if _calendar_year_from_path(f) in years]
    elif year_min is not None and year_max is not None:
        files = [f for f in files if (y := _calendar_year_from_path(f)) is not None and year_min <= y <= year_max]
    files.sort(key=lambda p: p.stem if p.stem.isdigit() else str(p))
    if max_daily_files is not None and max_daily_files > 0:
        files = files[:max_daily_files]
    return files


def _pick_dataarray(ds: Any, candidates: list[str]) -> Any:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"变量缺失，候选={candidates}，data_vars={list(ds.data_vars)}")


def _to_time_series_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1).mean(axis=1).astype(np.float32)


def _extract_wind_wave_series(nc_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    ds, tmp_copy = open_netcdf_dataset(nc_path)
    try:
        u_da = _pick_dataarray(ds, ["u10", "U10", "10u", "uwnd", "u_wind"])
        v_da = _pick_dataarray(ds, ["v10", "V10", "10v", "vwnd", "v_wind"])
        swh_da = _pick_dataarray(ds, ["swh", "SWH", "hs", "wave_height", "significant_wave_height"])
        u = _to_time_series_1d(u_da.values)
        v = _to_time_series_1d(v_da.values)
        swh = _to_time_series_1d(swh_da.values)
        t = min(len(u), len(v), len(swh))
        u, v, swh = u[:t], v[:t], swh[:t]
        wind = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2).astype(np.float32)
        wave = swh.astype(np.float32)
        feat = np.stack([wind, wave], axis=-1)
        bad = int(np.sum(~np.isfinite(feat)))
        if bad:
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feat, {"non_finite_replaced": bad, "time_steps": int(t)}
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


def _build_windows(ts: np.ndarray, window_steps: int, horizon_steps: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    t = int(ts.shape[0])
    need = window_steps + horizon_steps
    if t < need:
        raise ValueError(f"时间长度 T={t} 小于窗口需求 {need}（window={window_steps}, horizon={horizon_steps}）")
    starts = list(range(0, t - need + 1, stride))
    n = len(starts)
    x = np.empty((n, window_steps, 2), dtype=np.float32)
    y = np.empty((n, 2), dtype=np.float32)
    target_off = window_steps + horizon_steps - 1
    for i, s in enumerate(starts):
        x[i] = ts[s : s + window_steps]
        y[i] = ts[s + target_off]
    return x, y


def _save_split(path: str, x: np.ndarray, y: np.ndarray) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=x, y=y)
    print(f"wrote {p} X={x.shape} y={y.shape}")


def _concat_series(nc_files: list[Path]) -> tuple[np.ndarray, dict[str, int]]:
    if not nc_files:
        return np.empty((0, 2), dtype=np.float32), {"files_used": 0, "non_finite_replaced": 0}
    arrs: list[np.ndarray] = []
    meta = {"files_used": 0, "non_finite_replaced": 0}
    for fp in nc_files:
        feat, m = _extract_wind_wave_series(fp)
        arrs.append(feat)
        meta["files_used"] += 1
        meta["non_finite_replaced"] += int(m.get("non_finite_replaced", 0))
    return np.concatenate(arrs, axis=0), meta


def build_anomaly_from_netcdf(
    anomaly_cfg_path: str | Path,
    data_cfg_path: str | Path,
    max_daily_files: int | None,
    stride: int,
    proposition_year_split: bool | None = None,
) -> None:
    cfg = load_yaml(anomaly_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    raw_root = resolve_path(data_cfg["paths"]["raw_root"])
    pre = data_cfg.get("anomaly_preprocess") or {}
    subdir = pre.get("subdir") or data_cfg["paths"].get("anomaly_subdir") or "风-浪异常识别"

    # 风浪通常为 3-hourly；window/horizon 以小时配置，预处理转换为步数
    step_hours = int(pre.get("time_step_hours", 3))
    window_hours = int(cfg["data"].get("window_hours", 48))
    horizon_hours = int(cfg["data"].get("horizon_hours", 1))
    window_steps = max(1, window_hours // max(step_hours, 1))
    horizon_steps = max(1, (horizon_hours + step_hours - 1) // max(step_hours, 1))

    ysplit = data_cfg.get("anomaly_year_split") or {}
    use_prop = ysplit.get("enabled", True) if proposition_year_split is None else proposition_year_split

    paths = cfg["paths"]
    meta: dict[str, Any] = {
        "source": "netcdf",
        "window_hours": window_hours,
        "horizon_hours": horizon_hours,
        "time_step_hours": step_hours,
        "window_steps": window_steps,
        "horizon_steps": horizon_steps,
        "stride": stride,
        "split_mode": "proposition_years" if use_prop else "ratio",
    }

    if use_prop:
        tr = ysplit.get("train") or {}
        y_min = int(tr.get("min_year", 2014))
        y_max = int(tr.get("max_year", 2023))
        val_years = set(int(x) for x in ysplit.get("val_years", [2025]))
        test_years = set(int(x) for x in ysplit.get("test_years", [2024]))
        tr_files = discover_anomaly_nc_paths(raw_root, subdir, max_daily_files=max_daily_files, year_min=y_min, year_max=y_max)
        va_files = discover_anomaly_nc_paths(raw_root, subdir, years=val_years)
        te_files = discover_anomaly_nc_paths(raw_root, subdir, years=test_years)
        print(
            f"风浪年份划分: train {y_min}-{y_max}={len(tr_files)} 文件, "
            f"val {sorted(val_years)}={len(va_files)}, test {sorted(test_years)}={len(te_files)}",
            flush=True,
        )
        caps = {
            "train": pre.get("max_train_daily_files"),
            "val": pre.get("max_val_daily_files"),
            "test": pre.get("max_test_daily_files"),
        }
        split_files = {"train": tr_files, "val": va_files, "test": te_files}
        for split in ("train", "val", "test"):
            cap = caps[split]
            if cap is not None and int(cap) > 0 and len(split_files[split]) > int(cap):
                print(f"按 anomaly_preprocess.max_{split}_daily_files={cap} 截断 {split} 文件", flush=True)
                split_files[split] = split_files[split][: int(cap)]

        key_map = {"train": "train_sequences", "val": "val_sequences", "test": "test_sequences"}
        for split in ("train", "val", "test"):
            files = split_files[split]
            if not files:
                print(f"警告: {split} 无文件，跳过", file=sys.stderr)
                continue
            series, m = _concat_series(files)
            x, y = _build_windows(series, window_steps, horizon_steps, stride)
            _save_split(paths[key_map[split]], x, y)
            meta[f"{split}_files_used"] = int(m["files_used"])
            meta[f"{split}_non_finite_replaced"] = int(m["non_finite_replaced"])
            meta[f"{split}_samples"] = int(x.shape[0])
    else:
        all_files = discover_anomaly_nc_paths(raw_root, subdir, max_daily_files=max_daily_files)
        series, m = _concat_series(all_files)
        x, y = _build_windows(series, window_steps, horizon_steps, stride)
        sp = data_cfg.get("split") or {}
        tr = float(sp.get("train_ratio", 0.8))
        va = float(sp.get("val_ratio", 0.1))
        n = x.shape[0]
        i1 = int(n * tr)
        i2 = int(n * (tr + va))
        _save_split(paths["train_sequences"], x[:i1], y[:i1])
        _save_split(paths["val_sequences"], x[i1:i2], y[i1:i2])
        _save_split(paths["test_sequences"], x[i2:], y[i2:])
        meta["files_used"] = int(m["files_used"])
        meta["non_finite_replaced"] = int(m["non_finite_replaced"])
        meta["samples"] = int(n)

    tip = resolve_path(paths["typhoon_index"])
    tip.parent.mkdir(parents=True, exist_ok=True)
    with tip.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote {tip}")


def main() -> None:
    parser = argparse.ArgumentParser(description="风浪样本构建")
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument("--data-config", type=str, default="config/data.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--from-nc", action="store_true", help="从命题方风浪 NetCDF 构建 train/val/test.npz")
    parser.add_argument(
        "--max-daily-files",
        type=int,
        default=None,
        help="最多使用多少个日文件（默认读 data.yaml 的 anomaly_preprocess.max_daily_files）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="滑窗步长（默认读 data.yaml 的 anomaly_preprocess.window_stride）",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--year-split", action="store_true")
    split_group.add_argument("--ratio-split", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        generate_synthetic_anomaly(project_root() / args.config)
        return
    if args.from_nc:
        dcfg = load_yaml(args.data_config)
        pre = dcfg.get("anomaly_preprocess") or {}
        max_f = args.max_daily_files
        if max_f is None:
            mf = pre.get("max_daily_files")
            max_f = int(mf) if mf is not None else None
        stride = int(args.stride) if args.stride is not None else int(pre.get("window_stride", 1))
        prop_mode: bool | None = None
        if args.year_split:
            prop_mode = True
        elif args.ratio_split:
            prop_mode = False
        build_anomaly_from_netcdf(
            project_root() / args.config,
            project_root() / args.data_config,
            max_daily_files=max_f,
            stride=stride,
            proposition_year_split=prop_mode,
        )
        return
    raise SystemExit("请指定 --synthetic 或 --from-nc")


if __name__ == "__main__":
    main()
