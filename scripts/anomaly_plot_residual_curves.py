#!/usr/bin/env python3
"""
风—浪：连续时序「观测 vs 一步预测」与残差曲线（论文图 7-3）。

推荐（真实 NC，连续曲线）：
  python scripts/anomaly_plot_residual_curves.py --nc 服创数据集/风-浪异常识别/2024/20240101.nc

NPZ 分通道单样本（仅作调试；合成数据会出现负风速，不宜入论文）：
  python scripts/anomaly_plot_residual_curves.py --split test --n-plots 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.dataset import AnomalyNpzDataset
from src.anomaly.model import build_model
from src.preprocess.anomaly_dataset import (
    concat_wind_wave_year,
    extract_wind_wave_from_month_dir,
    extract_wind_wave_series_from_netcdf,
)
from src.utils.config import load_yaml, pick_device, project_root, resolve_path


def _setup_matplotlib() -> None:
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"):
        try:
            if any(name in (f.name or "") for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
        except Exception:
            pass


def _load_model(cfg: dict, ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _window_steps_from_cfg(cfg: dict, data_cfg: dict | None) -> tuple[int, int, int]:
    pre = (data_cfg or {}).get("anomaly_preprocess") or {}
    step_h = int(pre.get("time_step_hours", 3))
    win_h = int(cfg["data"].get("window_hours", 48))
    hor_h = int(cfg["data"].get("horizon_hours", 1))
    w = max(1, win_h // max(step_h, 1))
    h = max(1, (hor_h + step_h - 1) // max(step_h, 1))
    return w, h, step_h


@torch.no_grad()
def _rolling_predict(
    model: torch.nn.Module,
    series: np.ndarray,
    *,
    window_steps: int,
    horizon_steps: int,
    device: torch.device,
    plot_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """series (T,2) → 对齐到每个预测时刻的 obs、pred、残差；仅覆盖可滑窗段。"""
    ts = np.asarray(series, dtype=np.float32)
    t_len = int(ts.shape[0])
    need = window_steps + horizon_steps
    if t_len < need:
        raise ValueError(f"序列长度 T={t_len} 不足窗口需求 {need}")

    starts = list(range(0, t_len - need + 1, max(1, plot_stride)))
    t_idx: list[int] = []
    obs: list[np.ndarray] = []
    prd: list[np.ndarray] = []
    for s in starts:
        tgt = s + window_steps + horizon_steps - 1
        x = torch.from_numpy(ts[s : s + window_steps]).float().unsqueeze(0).to(device)
        p = model(x).cpu().numpy()[0]
        t_idx.append(tgt)
        obs.append(ts[tgt])
        prd.append(p)
    t_arr = np.asarray(t_idx, dtype=np.int32)
    o_arr = np.stack(obs, axis=0)
    p_arr = np.stack(prd, axis=0)
    return t_arr, o_arr, p_arr, o_arr - p_arr


def _plot_continuous(
    *,
    t_idx: np.ndarray,
    obs: np.ndarray,
    pred: np.ndarray,
    resid: np.ndarray,
    full_series: np.ndarray | None,
    out_path: Path,
    title: str,
    step_hours: int,
) -> None:
    """三行：风速、波高、残差（双通道）。"""
    _setup_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.2]})
    labels = ("风速模长 |U10|", "有效波高 SWH")
    colors = ("#0369a1", "#b45309")

    if full_series is not None and full_series.shape[0] > 0:
        t_full = np.arange(full_series.shape[0]) * step_hours
        for ch in range(2):
            axes[ch].plot(t_full, full_series[:, ch], color=colors[ch], alpha=0.25, lw=1.0, label="全序列（背景）")

    t_h = t_idx * step_hours
    for ch in range(2):
        ax = axes[ch]
        ax.plot(t_h, obs[:, ch], "o-", color=colors[ch], ms=3, lw=1.2, label="观测")
        ax.plot(t_h, pred[:, ch], "x--", color=colors[ch], ms=4, lw=1.0, alpha=0.85, label="LSTM 一步预测")
        ax.set_ylabel(labels[ch])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if ch == 0 and np.any(obs[:, 0] < 0):
            ax.axhline(0, color="#94a3b8", ls=":", lw=0.8)

    ax_r = axes[2]
    ax_r.plot(t_h, resid[:, 0], "-", color=colors[0], lw=1.0, label="残差·风")
    ax_r.plot(t_h, resid[:, 1], "-", color=colors[1], lw=1.0, label="残差·浪")
    ax_r.axhline(0, color="#64748b", lw=0.8)
    # 简易 3σ 带（段内）
    for ch, c in enumerate(colors):
        mu, sig = float(resid[:, ch].mean()), float(resid[:, ch].std()) + 1e-6
        ax_r.fill_between(t_h, mu - 3 * sig, mu + 3 * sig, color=c, alpha=0.08)
    ax_r.set_ylabel("残差 (观测−预测)")
    ax_r.set_xlabel(f"时间步索引 × {step_hours} h")
    ax_r.legend(fontsize=8, ncol=2)
    ax_r.grid(True, alpha=0.3)

    mae = np.abs(resid).mean(axis=0)
    fig.suptitle(f"{title}\n段内 MAE: 风={mae[0]:.3f} 浪={mae[1]:.3f}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_npz_sample(
    x_win: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    r: np.ndarray,
    out_path: Path,
    title: str,
    step_hours: int,
) -> None:
    """单样本：分通道显示历史窗 + 预测点（非连续多步）。"""
    _setup_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [1.5, 1.5, 1]})
    labels = ("风速模长", "有效波高")
    colors = ("#0369a1", "#b45309")
    t_hist = np.arange(x_win.shape[0]) * step_hours
    t_pred = x_win.shape[0] * step_hours

    for ch in range(2):
        ax = axes[ch]
        ax.plot(t_hist, x_win[:, ch], "-", color=colors[ch], lw=1.5, label="历史窗")
        ax.scatter([t_pred], [float(y[ch])], color=colors[ch], marker="o", s=55, zorder=4, label="真值")
        ax.scatter([t_pred], [float(p[ch])], color=colors[ch], marker="x", s=70, zorder=5, label="预测")
        ax.axvline(t_pred, color="#94a3b8", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylabel(labels[ch])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar([0, 1], r, color=colors, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("一步残差")
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _warn_data_quality(series: np.ndarray, source: str) -> None:
    neg_w = float((series[:, 0] < 0).mean()) if series.size else 0.0
    if neg_w > 0.05:
        print(
            f"警告 [{source}]：风速模长有 {neg_w * 100:.1f}% 为负值，"
            "通常表示合成随机数据或 NC 拼接/缺测处理问题；论文图请用 --from-nc 真实文件。",
            file=sys.stderr,
        )


def _load_series_from_args(
    args: argparse.Namespace,
    data_cfg: dict,
) -> tuple[np.ndarray, str, str]:
    """返回 (series, stem, source_label)。"""
    raw = resolve_path(data_cfg["paths"]["raw_root"])
    sub = (data_cfg.get("anomaly_preprocess") or {}).get("subdir") or "风浪异常识别"

    if args.year is not None:
        series, meta = concat_wind_wave_year(raw, sub, int(args.year))
        stem = f"year{args.year}"
        label = f"{args.year} 年（{len(meta.get('months', []))} 个月）"
        return series, stem, label

    if args.month_dir:
        md = resolve_path(args.month_dir)
        series, meta = extract_wind_wave_from_month_dir(md)
        stem = md.name
        try:
            label = str(md.relative_to(project_root().resolve()))
        except ValueError:
            label = str(md)
        return series, stem, label

    if args.nc:
        nc = resolve_path(args.nc)
        if nc.is_dir():
            series, _ = extract_wind_wave_from_month_dir(nc)
            stem = nc.name
            label = str(nc)
        else:
            series, _ = extract_wind_wave_series_from_netcdf(nc)
            stem = nc.stem
            label = nc.name
        return series, stem, label

    raise SystemExit("请指定 --year、--month-dir 或 --nc（命题方数据）")


@torch.no_grad()
def run_continuous(
    *,
    series: np.ndarray,
    stem: str,
    title_label: str,
    cfg: dict,
    data_cfg: dict,
    ckpt: Path,
    out_dir: Path,
    plot_stride: int,
    max_points: int,
) -> None:
    if series.shape[0] < 10:
        raise SystemExit(f"序列过短: {series.shape}")
    _warn_data_quality(series, title_label)

    window_steps, horizon_steps, step_h = _window_steps_from_cfg(cfg, data_cfg)
    device = torch.device(pick_device(cfg["train"].get("device", "cpu")))
    model = _load_model(cfg, ckpt, device)

    if max_points > 0:
        need = window_steps + horizon_steps + max_points * plot_stride
        if series.shape[0] > need:
            series = series[:need]

    t_idx, obs, pred, resid = _rolling_predict(
        model,
        series,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        device=device,
        plot_stride=plot_stride,
    )
    out = out_dir / f"{stem}_continuous.png"
    _plot_continuous(
        t_idx=t_idx,
        obs=obs,
        pred=pred,
        resid=resid,
        full_series=series,
        out_path=out,
        title=f"风—浪一步预测 · {title_label}",
        step_hours=step_h,
    )
    summary = out_dir / f"{stem}_continuous.txt"
    mae = np.abs(resid).mean(axis=0)
    summary.write_text(
        f"source={title_label}\nT={series.shape[0]} window={window_steps} horizon={horizon_steps}\n"
        f"plot_stride={plot_stride} n_points={len(t_idx)}\n"
        f"mae_wind={mae[0]:.6f} mae_wave={mae[1]:.6f}\n"
        f"wind_range=[{series[:,0].min():.3f},{series[:,0].max():.3f}] "
        f"wave_range=[{series[:,1].min():.3f},{series[:,1].max():.3f}]\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(summary.read_text(encoding="utf-8"))


@torch.no_grad()
def run_from_npz(
    *,
    cfg: dict,
    data_cfg: dict,
    ckpt: Path,
    split: str,
    out_dir: Path,
    n_plots: int,
    top_residual: int,
    seed: int,
) -> None:
    from torch.utils.data import DataLoader

    paths = cfg["paths"]
    ds = AnomalyNpzDataset(paths[f"{split}_sequences"])
    _warn_data_quality(ds.x.reshape(-1, 2), f"{split}.npz")

    loader = DataLoader(ds, batch_size=128, shuffle=False)
    device = torch.device(pick_device(cfg["train"].get("device", "cpu")))
    model = _load_model(cfg, ckpt, device)

    xs, ys, ps = [], [], []
    for x, y in loader:
        pred = model(x.to(device)).cpu().numpy()
        xs.append(x.numpy())
        ys.append(y.numpy())
        ps.append(pred)
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(ps, axis=0)
    resid = y_all - p_all
    n = y_all.shape[0]

    if top_residual > 0:
        score = np.abs(resid).mean(axis=1)
        order = np.argsort(-score)[: min(top_residual, n)]
    else:
        rng = np.random.default_rng(seed)
        order = rng.choice(n, size=min(n_plots, n), replace=False)

    _, _, step_h = _window_steps_from_cfg(cfg, data_cfg)
    sub = out_dir / split
    for j, i in enumerate(order):
        _plot_npz_sample(
            x_all[i],
            y_all[i],
            p_all[i],
            resid[i],
            sub / f"sample_{j:04d}_idx{i}.png",
            title=f"{split} 样本 #{int(i)}（单窗一步预测，非连续曲线）",
            step_hours=step_h,
        )
    print(f"wrote {len(order)} NPZ sample figures -> {sub}")


def main() -> None:
    ap = argparse.ArgumentParser(description="风—浪观测-预测-残差图")
    ap.add_argument("--config", default="config/anomaly.yaml")
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--ckpt", default="outputs/anomaly/best.pt")
    ap.add_argument("--nc", type=str, default=None, help="单个 NC 或月份目录（含 oper/wave 双文件）")
    ap.add_argument(
        "--month-dir",
        type=str,
        default=None,
        help="月份目录，如 服创数据集/风浪异常识别/2024/202401",
    )
    ap.add_argument("--year", type=int, default=None, help="按年拼接各月 oper+wave 后绘图（如 2024 测试年）")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--n-plots", type=int, default=0)
    ap.add_argument("--top-residual", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-stride", type=int, default=3, help="连续预测绘图步长（越大点越少）")
    ap.add_argument("--max-points", type=int, default=120, help="连续模式最多预测点数，0=不限制")
    ap.add_argument("--out", default="outputs/anomaly/figures")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"权重不存在: {ckpt}")
    out_dir = resolve_path(args.out)

    if args.nc or args.month_dir or args.year is not None:
        series, stem, label = _load_series_from_args(args, data_cfg)
        run_continuous(
            series=series,
            stem=stem,
            title_label=label,
            cfg=cfg,
            data_cfg=data_cfg,
            ckpt=ckpt,
            out_dir=out_dir,
            plot_stride=max(1, args.plot_stride),
            max_points=args.max_points,
        )
        return

    n = args.n_plots if args.n_plots > 0 else 4
    run_from_npz(
        cfg=cfg,
        data_cfg=data_cfg,
        ckpt=ckpt,
        split=args.split,
        out_dir=out_dir / "residual_cases",
        n_plots=n,
        top_residual=args.top_residual,
        seed=args.seed,
    )
    print(
        "提示：论文图请用命题方数据，例如：\n"
        "  python scripts/anomaly_plot_residual_curves.py --month-dir 服创数据集/风浪异常识别/2024/202401\n"
        "  python scripts/anomaly_plot_residual_curves.py --year 2024 --plot-stride 6",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
