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


def _pick_dataarray(ds: Any, candidates: list[str], *, required: bool = True) -> Any | None:
    for name in candidates:
        if name in ds:
            return ds[name]
    if required:
        raise KeyError(f"变量缺失，候选={candidates}，data_vars={list(ds.data_vars)}")
    return None


def _to_time_series_1d(arr: np.ndarray) -> np.ndarray:
    """时间维保留，其余维 nanmean（格点常有海陆掩膜，普通 mean 会得到全 NaN）。"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.nanmean(flat, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _extract_wind_wave_series(nc_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    ds, tmp_copy = open_netcdf_dataset(nc_path)
    try:
        u_da = _pick_dataarray(ds, ["u10", "U10", "10u", "uwnd", "u_wind"], required=False)
        v_da = _pick_dataarray(ds, ["v10", "V10", "10v", "vwnd", "v_wind"], required=False)
        swh_da = _pick_dataarray(
            ds, ["swh", "SWH", "hs", "wave_height", "significant_wave_height"], required=False
        )
        has_uv = u_da is not None and v_da is not None
        has_swh = swh_da is not None
        if not has_uv and not has_swh:
            raise KeyError(
                f"变量缺失，需至少包含 [u10,v10] 或 [swh]。data_vars={list(ds.data_vars)}"
            )
        if has_uv:
            u = _to_time_series_1d(u_da.values)  # type: ignore[union-attr]
            v = _to_time_series_1d(v_da.values)  # type: ignore[union-attr]
            wind = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2).astype(np.float32)
        else:
            wind = np.empty((0,), dtype=np.float32)
        if has_swh:
            wave = _to_time_series_1d(swh_da.values).astype(np.float32)  # type: ignore[union-attr]
            used_wave_fallback = 0
        elif has_uv:
            wave = wind.copy()
            used_wave_fallback = 1
        else:
            wave = np.empty((0,), dtype=np.float32)

        if wind.size and wave.size:
            t = min(len(wind), len(wave))
            feat = np.stack([wind[:t], wave[:t]], axis=-1)
        elif wind.size:
            feat = np.stack([wind, np.full_like(wind, np.nan)], axis=-1)
        elif wave.size:
            feat = np.stack([np.full_like(wave, np.nan), wave], axis=-1)
        else:
            feat = np.empty((0, 2), dtype=np.float32)

        bad = int(np.sum(~np.isfinite(feat)))
        if bad:
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feat, {
            "non_finite_replaced": bad,
            "time_steps": int(feat.shape[0]),
            "used_wave_fallback": used_wave_fallback,
            "has_wind": int(has_uv),
            "has_wave": int(has_swh),
        }
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


def extract_wind_wave_series_from_netcdf(nc_path: str | Path) -> tuple[np.ndarray, dict[str, int]]:
    """从 NetCDF 提取 (T,2) 时序：[风速模长, 浪高]；变量约定与 `_extract_wind_wave_series` 一致。"""
    return _extract_wind_wave_series(Path(nc_path))


def _spatial_mean_ts(da: Any) -> np.ndarray:
    """时间维保留，其余维 nanmean（命题方风浪格点常有海陆掩膜）。"""
    arr = np.asarray(da.values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    t = arr.shape[0]
    flat = arr.reshape(t, -1)
    out = np.nanmean(flat, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_wind_wave_from_month_dir(month_dir: str | Path) -> tuple[np.ndarray, dict[str, int]]:
    """
    「风-浪异常识别」按月目录：通常含
    `data_stream-oper_stepType-instant.nc`（u10/v10）与
    `data_stream-wave_stepType-instant.nc`（swh）。
    返回对齐后的 (T,2)：[|U10|, SWH]。
    """
    md = Path(month_dir)
    if not md.is_dir():
        raise FileNotFoundError(f"月份目录不存在: {md}")

    oper_candidates = sorted(md.glob("*oper*.nc"))
    wave_candidates = sorted(md.glob("*wave*.nc"))
    if not oper_candidates and not wave_candidates:
        raise FileNotFoundError(f"未找到 oper/wave NC: {md}")

    wind = np.empty((0,), dtype=np.float32)
    wave = np.empty((0,), dtype=np.float32)
    meta: dict[str, int] = {"has_wind": 0, "has_wave": 0, "used_wave_fallback": 0}

    if oper_candidates:
        ds, tmp = open_netcdf_dataset(oper_candidates[0])
        try:
            u_da = _pick_dataarray(ds, ["u10", "U10", "10u", "uwnd", "u_wind"], required=False)
            v_da = _pick_dataarray(ds, ["v10", "V10", "10v", "vwnd", "v_wind"], required=False)
            if u_da is not None and v_da is not None:
                u = _spatial_mean_ts(u_da)
                v = _spatial_mean_ts(v_da)
                wind = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2).astype(np.float32)
                meta["has_wind"] = 1
        finally:
            ds.close()
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                except OSError:
                    pass

    if wave_candidates:
        ds, tmp = open_netcdf_dataset(wave_candidates[0])
        try:
            swh_da = _pick_dataarray(
                ds, ["swh", "SWH", "hs", "wave_height", "significant_wave_height"], required=False
            )
            if swh_da is not None:
                wave = _spatial_mean_ts(swh_da)
                meta["has_wave"] = 1
        finally:
            ds.close()
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                except OSError:
                    pass

    if wind.size and wave.size:
        t = min(len(wind), len(wave))
        feat = np.stack([wind[:t], wave[:t]], axis=-1)
    elif wind.size:
        meta["used_wave_fallback"] = 1
        feat = np.stack([wind, wind.copy()], axis=-1)
    elif wave.size:
        feat = np.stack([wave.copy(), wave], axis=-1)
        meta["used_wave_fallback"] = 1
    else:
        raise KeyError(f"月份目录无可用风/浪变量: {md}")

    meta["time_steps"] = int(feat.shape[0])
    meta["month_dir"] = str(md)
    return feat, meta


def discover_anomaly_month_dirs(
    raw_root: Path,
    subdir: str,
    *,
    years: set[int] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[Path]:
    """命题方目录：风浪异常识别/{年}/{月}/ 下 oper+wave 配对。"""
    root = raw_root / subdir
    if not root.is_dir():
        raise FileNotFoundError(f"风浪目录不存在: {root}")
    out: list[Path] = []
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        y = int(year_dir.name)
        if years is not None and y not in years:
            continue
        if year_min is not None and year_max is not None and not (year_min <= y <= year_max):
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if month_dir.is_dir():
                out.append(month_dir)
    return out


def _concat_month_dirs(month_dirs: list[Path]) -> tuple[np.ndarray, dict[str, Any]]:
    parts: list[np.ndarray] = []
    meta: dict[str, Any] = {"months_used": 0, "months_skipped": 0}
    for md in month_dirs:
        try:
            feat, _ = extract_wind_wave_from_month_dir(md)
        except (FileNotFoundError, KeyError) as e:
            print(f"警告: 跳过 {md.name}: {e}", file=sys.stderr)
            meta["months_skipped"] = int(meta["months_skipped"]) + 1
            continue
        if feat.shape[0] > 0:
            parts.append(feat)
            meta["months_used"] = int(meta["months_used"]) + 1
    if not parts:
        return np.empty((0, 2), dtype=np.float32), meta
    cat = np.concatenate(parts, axis=0)
    meta["T"] = int(cat.shape[0])
    return cat, meta


def concat_wind_wave_year(raw_root: Path, subdir: str, year: int) -> tuple[np.ndarray, dict[str, Any]]:
    """按年拼接各月 oper+wave 对齐序列（用于连续绘图/抽查）。"""
    root = raw_root / subdir / str(year)
    if not root.is_dir():
        raise FileNotFoundError(f"年份目录不存在: {root}")
    month_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    parts: list[np.ndarray] = []
    used_months: list[str] = []
    for md in month_dirs:
        try:
            feat, _ = extract_wind_wave_from_month_dir(md)
        except (FileNotFoundError, KeyError) as e:
            print(f"警告: 跳过 {md.name}: {e}", file=sys.stderr)
            continue
        if feat.shape[0] > 0:
            parts.append(feat)
            used_months.append(md.name)
    if not parts:
        return np.empty((0, 2), dtype=np.float32), {"year": year, "months": []}
    cat = np.concatenate(parts, axis=0)
    return cat, {"year": year, "months": used_months, "T": int(cat.shape[0])}


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
    wind_parts: list[np.ndarray] = []
    wave_parts: list[np.ndarray] = []
    meta = {
        "files_used": 0,
        "non_finite_replaced": 0,
        "files_skipped": 0,
        "used_wave_fallback": 0,
        "wind_only_files": 0,
        "wave_only_files": 0,
        "both_vars_files": 0,
    }
    for fp in nc_files:
        try:
            feat, m = _extract_wind_wave_series(fp)
        except KeyError as e:
            print(f"警告: 跳过变量不完整文件 {fp.name}: {e}", file=sys.stderr)
            meta["files_skipped"] += 1
            continue
        meta["files_used"] += 1
        meta["non_finite_replaced"] += int(m.get("non_finite_replaced", 0))
        meta["used_wave_fallback"] += int(m.get("used_wave_fallback", 0))
        has_wind = int(m.get("has_wind", 0)) == 1
        has_wave = int(m.get("has_wave", 0)) == 1
        if has_wind and has_wave:
            meta["both_vars_files"] += 1
        elif has_wind:
            meta["wind_only_files"] += 1
        elif has_wave:
            meta["wave_only_files"] += 1
        if feat.shape[0] == 0:
            continue
        if has_wind:
            wind_parts.append(feat[:, 0].astype(np.float32))
        if has_wave:
            wave_parts.append(feat[:, 1].astype(np.float32))

    if not wind_parts and not wave_parts:
        return np.empty((0, 2), dtype=np.float32), meta
    if wind_parts:
        wind = np.concatenate(wind_parts, axis=0)
    else:
        wind = np.empty((0,), dtype=np.float32)
    if wave_parts:
        wave = np.concatenate(wave_parts, axis=0)
    else:
        wave = np.empty((0,), dtype=np.float32)
    if wind.size == 0 and wave.size > 0:
        wind = wave.copy()
        meta["used_wave_fallback"] += 1
    if wave.size == 0 and wind.size > 0:
        wave = wind.copy()
        meta["used_wave_fallback"] += 1
    t = min(wind.size, wave.size)
    if t == 0:
        return np.empty((0, 2), dtype=np.float32), meta
    ts = np.stack([wind[:t], wave[:t]], axis=-1).astype(np.float32)
    return ts, meta


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
        meta["merge_mode"] = "month_oper_wave"
        tr_months = discover_anomaly_month_dirs(raw_root, subdir, year_min=y_min, year_max=y_max)
        va_months = discover_anomaly_month_dirs(raw_root, subdir, years=val_years)
        te_months = discover_anomaly_month_dirs(raw_root, subdir, years=test_years)
        print(
            f"风浪年份划分（按月 oper+wave 合并）: train {y_min}-{y_max}={len(tr_months)} 月, "
            f"val {sorted(val_years)}={len(va_months)} 月, test {sorted(test_years)}={len(te_months)} 月",
            flush=True,
        )
        caps = {
            "train": pre.get("max_train_daily_files"),
            "val": pre.get("max_val_daily_files"),
            "test": pre.get("max_test_daily_files"),
        }
        split_months = {"train": tr_months, "val": va_months, "test": te_months}
        for split in ("train", "val", "test"):
            cap = caps[split]
            if cap is not None and int(cap) > 0 and len(split_months[split]) > int(cap):
                print(f"按 max_{split}_daily_files 截断为前 {cap} 个月目录", flush=True)
                split_months[split] = split_months[split][: int(cap)]

        key_map = {"train": "train_sequences", "val": "val_sequences", "test": "test_sequences"}
        for split in ("train", "val", "test"):
            months = split_months[split]
            if not months:
                print(f"警告: {split} 无月份目录，跳过", file=sys.stderr)
                continue
            series, m = _concat_month_dirs(months)
            if series.shape[0] < window_steps + horizon_steps:
                print(f"警告: {split} 序列过短 T={series.shape[0]}，跳过", file=sys.stderr)
                continue
            x, y = _build_windows(series, window_steps, horizon_steps, stride)
            _save_split(paths[key_map[split]], x, y)
            meta[f"{split}_months_used"] = int(m.get("months_used", 0))
            meta[f"{split}_months_skipped"] = int(m.get("months_skipped", 0))
            meta[f"{split}_samples"] = int(x.shape[0])
            meta[f"{split}_T"] = int(m.get("T", 0))
            wy = float(y[:, 1].max()) if y.shape[0] else 0.0
            if wy <= 1e-6:
                print(f"警告: {split} 标签波高全为 0，请检查月份目录是否含 wave NC", file=sys.stderr)
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
        meta["files_skipped"] = int(m.get("files_skipped", 0))
        meta["non_finite_replaced"] = int(m["non_finite_replaced"])
        meta["used_wave_fallback"] = int(m.get("used_wave_fallback", 0))
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
