from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.preprocess.hydro_nc_stack import (
    apply_zscore,
    build_windows,
    discover_hydro_nc_paths,
    split_train_val_test,
    stack_hydro_fields,
    zscore_fit,
)
from src.utils.config import load_yaml, project_root, resolve_path, ensure_dir


def generate_synthetic(config_path: str | Path) -> None:
    """写入 X_*.npz / y_*.npz，形状与 config 中 input_steps、output_steps、grid_shape、特征数一致。"""
    cfg = load_yaml(config_path)
    root = project_root()
    paths = cfg["paths"]
    d = cfg["data"]
    tin = int(d["input_steps"])
    tout = int(d["output_steps"])
    gh, gw = int(d["grid_shape"][0]), int(d["grid_shape"][1])
    c_in = len(d["input_features"])
    c_out = len(d["target_features"])
    rng = np.random.default_rng(int(cfg["meta"]["seed"]))

    n_train, n_val, n_test = 80, 16, 16

    def make_batch(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.standard_normal((n, tin, gh, gw, c_in)).astype(np.float32)
        y = rng.standard_normal((n, tout, gh, gw, c_out)).astype(np.float32)
        y += 0.05 * x[:, -tout:, :, :, :c_out]
        return x, y

    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        x, y = make_batch(n)
        xp = resolve_path(paths[f"{name}_data"])
        yp = resolve_path(paths[f"{name}_label"])
        xp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(xp, X=x)
        np.savez_compressed(yp, y=y)
        print(f"wrote {xp.name} / {yp.name} shape X={x.shape} y={y.shape}")


def build_from_netcdf(
    hydro_cfg_path: Path,
    data_cfg_path: Path,
    max_daily_files: int | None,
    window_stride: int,
) -> None:
    """从服创数据集/海域要素预测 读取多日 .nc，滑窗、划分、Z-score，写入 data/processed/hydro/。"""
    hydro = load_yaml(hydro_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    root = project_root()

    raw_root = resolve_path(data_cfg["paths"]["raw_root"])
    hydro_sub = data_cfg["paths"].get("hydro_subdir") or "海域要素预测"

    d = hydro["data"]
    feats = list(d["input_features"])
    if set(d["target_features"]) != set(feats) or len(d["target_features"]) != len(feats):
        raise NotImplementedError("当前仅支持 input_features 与 target_features 一致（四要素）")
    tin = int(d["input_steps"])
    tout = int(d["output_steps"])

    nc_list = discover_hydro_nc_paths(raw_root, hydro_sub, max_daily_files=max_daily_files)
    print(f"使用 NetCDF 文件数: {len(nc_list)}（max_daily_files={max_daily_files}）")
    if max_daily_files is None and len(nc_list) > 500:
        print(
            "警告: 未限制日文件数量，数据量可能极大、预处理耗时与内存占用高；"
            "建议先设 --max-daily-files 或 data.yaml 中 hydro_preprocess.max_daily_files。",
            file=sys.stderr,
        )

    field, meta = stack_hydro_fields(nc_list, feats)
    print(f"拼接后场 shape (T,H,W,C)={field.shape}, meta={meta}")

    x, y = build_windows(field, tin, tout, stride=window_stride)
    print(f"滑窗样本数 N={x.shape[0]}, stride={window_stride}")

    sp = data_cfg.get("split", {})
    tr, va, te = float(sp.get("train_ratio", 0.8)), float(sp.get("val_ratio", 0.1)), float(sp.get("test_ratio", 0.1))
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = split_train_val_test(x, y, tr, va, te)

    mean, std = zscore_fit(x_tr)
    x_tr = apply_zscore(x_tr, mean, std)
    x_va = apply_zscore(x_va, mean, std)
    x_te = apply_zscore(x_te, mean, std)
    y_tr = apply_zscore(y_tr, mean, std)
    y_va = apply_zscore(y_va, mean, std)
    y_te = apply_zscore(y_te, mean, std)

    paths = hydro["paths"]
    stats_dir = resolve_path(data_cfg.get("normalization", {}).get("stats_dir", "data/processed/stats"))
    ensure_dir(stats_dir)
    stats_npz = stats_dir / "hydro_zscore.npz"
    np.savez_compressed(stats_npz, mean=mean, std=std, features=feats)
    with (stats_dir / "hydro_preprocess_meta.json").open("w", encoding="utf-8") as f:
        json.dump({**meta, "hydro_config": str(hydro_cfg_path), "window_stride": window_stride}, f, ensure_ascii=False, indent=2)

    for split_name, x_arr, y_arr in (
        ("train", x_tr, y_tr),
        ("val", x_va, y_va),
        ("test", x_te, y_te),
    ):
        xp = resolve_path(paths[f"{split_name}_data"])
        yp = resolve_path(paths[f"{split_name}_label"])
        ensure_dir(xp.parent)
        np.savez_compressed(xp, X=x_arr)
        np.savez_compressed(yp, y=y_arr)
        print(f"wrote {xp} / {yp} shapes X={x_arr.shape} y={y_arr.shape}")

    print(f"统计量: {stats_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(description="水文样本构建")
    parser.add_argument("--config", type=str, default="config/hydro.yaml", help="水文 YAML（hydro_hycom 用四要素）")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="生成与 config 一致的合成 npz（无命题方数据时使用）",
    )
    parser.add_argument(
        "--from-nc",
        action="store_true",
        help="从 config/data.yaml 的 raw_root + 海域要素预测 读取 NetCDF 并生成 npz",
    )
    parser.add_argument("--data-config", type=str, default="config/data.yaml")
    parser.add_argument(
        "--max-daily-files",
        type=int,
        default=None,
        help="最多使用多少个日文件（默认读 data.yaml 的 hydro_preprocess.max_daily_files）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="滑窗步长（默认读 data.yaml 的 hydro_preprocess.window_stride）",
    )
    args = parser.parse_args()
    root = project_root()

    if args.synthetic:
        generate_synthetic(root / args.config)
        return

    if args.from_nc:
        data_cfg = load_yaml(args.data_config)
        hp = data_cfg.get("hydro_preprocess") or {}
        max_f = args.max_daily_files
        if max_f is None:
            mf = hp.get("max_daily_files")
            max_f = int(mf) if mf is not None else None
        stride = args.stride
        if stride is None:
            stride = int(hp.get("window_stride", 1))

        build_from_netcdf(
            root / args.config,
            root / args.data_config,
            max_daily_files=max_f,
            window_stride=stride,
        )
        return

    raise SystemExit(
        "请指定 --synthetic 或 --from-nc；命题方 NetCDF 示例：\n"
        "  python -m src.preprocess.hydro_dataset --config config/hydro_hycom.yaml "
        "--from-nc --data-config config/data.yaml --max-daily-files 120"
    )


if __name__ == "__main__":
    main()
