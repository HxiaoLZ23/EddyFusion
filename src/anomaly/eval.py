"""
模块 C 回归评估（层 A：风浪一步预测 MAE/RMSE）。

- **主指标**：`mae_wind` / `mae_wave` / `mae_avg` 等，用于论文与材料表。
- **`passed`**：`mae_avg < 0.5` 仅为工程占位，**不是**赛题「真实台风识别比例」。
- **台风关联 Recall**：见 `scripts/anomaly_typhoon_link_eval.py` 与
  `docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md`（层 B，不重训）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python src/anomaly/eval.py` 直接运行时导入 `src.*`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly.dataset import AnomalyNpzDataset
from src.anomaly.model import build_model
from src.utils.config import load_yaml, pick_device, resolve_path
from src.utils.metrics import write_metrics_json


def _window_horizon_steps(cfg: dict, data_cfg: dict | None = None) -> tuple[int, int, int]:
    pre = (data_cfg or {}).get("anomaly_preprocess") or {}
    step_h = int(pre.get("time_step_hours", 3))
    win_h = int(cfg["data"].get("window_hours", 48))
    hor_h = int(cfg["data"].get("horizon_hours", 1))
    w = max(1, win_h // max(step_h, 1))
    h = max(1, (hor_h + step_h - 1) // max(step_h, 1))
    return w, h, step_h


def _label_and_persistence_stats(ds: AnomalyNpzDataset) -> dict[str, float]:
    """NPZ 标签物理量纲统计 + 持续性基线（用窗口末步预测下一目标）。"""
    y = np.asarray(ds.y, dtype=np.float64)
    x_last = np.asarray(ds.x[:, -1, :], dtype=np.float64)
    pers_mae = np.abs(x_last - y).mean(axis=0)
    out: dict[str, float] = {}
    for i, name in enumerate(("wind", "wave")):
        col = y[:, i]
        out[f"label_{name}_mean"] = float(col.mean())
        out[f"label_{name}_std"] = float(col.std())
        out[f"label_{name}_min"] = float(col.min())
        out[f"label_{name}_max"] = float(col.max())
        out[f"persistence_mae_{name}"] = float(pers_mae[i])
    out["persistence_mae_avg"] = float(pers_mae.mean())
    return out


@torch.no_grad()
def run_eval(cfg: dict, ckpt: Path, device: torch.device, split: str, *, data_cfg: dict | None = None) -> dict:
    paths = cfg["paths"]
    key = f"{split}_sequences"
    if key not in paths:
        raise KeyError(f"配置 paths 中缺少 {key}，无法评估 split={split}")
    ds = AnomalyNpzDataset(paths[key])
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    label_stats = _label_and_persistence_stats(ds)
    window_steps, horizon_steps, step_hours = _window_horizon_steps(cfg, data_cfg)

    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    maes = []
    rmses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        mae = (pred - y).abs().mean(dim=0)
        rmse = torch.sqrt(((pred - y) ** 2).mean(dim=0))
        maes.append(mae.cpu().numpy())
        rmses.append(rmse.cpu().numpy())
    mae = np.stack(maes, axis=0).mean(0)
    rmse = np.stack(rmses, axis=0).mean(0)
    mae_avg = float(mae.mean())
    pers_avg = label_stats["persistence_mae_avg"]
    metrics = {
        "mae_wind": float(mae[0]),
        "mae_wave": float(mae[1]),
        "rmse_wind": float(rmse[0]),
        "rmse_wave": float(rmse[1]),
        "mae_avg": mae_avg,
        "rmse_avg": float(rmse.mean()),
        "split": split,
        **label_stats,
        "mae_avg_vs_persistence_ratio": float(mae_avg / max(pers_avg, 1e-9)),
        "window_steps": int(window_steps),
        "horizon_steps": int(horizon_steps),
        "time_step_hours": int(step_hours),
        "target_units": "wind_mps_wave_m",
        "normalization": "none",
        "eval_note": (
            "MAE/RMSE 在原始物理量纲上计算（|U10| m/s、SWH m）；预处理与 eval 均无 StandardScaler。"
            f"任务为区域空间平均序列 {horizon_steps} 步（{horizon_steps * step_hours}h）超前；"
            "数值偏小主因序列平滑+短超前，需对照 persistence_mae_* 与 label_*_mean。"
        ),
    }
    if split == "val":
        metrics["val_mae_avg"] = metrics["mae_avg"]
    else:
        metrics["test_mae_avg"] = metrics["mae_avg"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/anomaly/best.pt")
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default="val",
        help="使用 paths 中 val_sequences 或 test_sequences",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml("config/data.yaml")
    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        out = resolve_path(cfg["paths"]["output_dir"])
        last = out / "last.pt"
        hint = f"未找到权重: {ckpt}"
        if last.is_file() and ckpt.name == "best.pt":
            hint += f"\n可改用: --ckpt {last}"
        raise FileNotFoundError(hint)

    device = torch.device(pick_device(cfg["train"].get("device", "cuda")))
    metrics = run_eval(cfg, ckpt, device, split=args.split, data_cfg=data_cfg)

    level = int(cfg["meta"]["level"])
    # passed：工程占位（mae_avg<0.5），勿写入论文为「台风识别准确率≥80%」
    passed = metrics["mae_avg"] < 0.5

    mf = cfg.get("eval", {}).get("metrics_file", "outputs/anomaly/metrics_summary.json")
    mp = resolve_path(mf)
    out_json = mp.parent / f"{mp.stem}_{args.split}{mp.suffix}"
    write_metrics_json(
        out_json,
        module="anomaly",
        level=level,
        metrics=metrics,
        passed=passed,
        tags={"level": level, "eval_split": args.split},
    )
    print(f"wrote {out_json}")
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
