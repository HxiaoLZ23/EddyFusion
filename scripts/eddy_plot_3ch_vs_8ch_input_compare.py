#!/usr/bin/env python3
"""图 3-3 / 图 4-2：3ch RGB vs 7ch（或 8ch）物理堆叠 — 指标柱 + val 预测拼图（统一 conf）。

布局：上行 mask mAP@0.5；下行左右为 **同一 conf 阈值** 下本地重绘的 val batch 拼图。

示例（定稿 3ch vs 7ch，指标与表 5-4 一致）::

  python scripts/eddy_plot_3ch_vs_8ch_input_compare.py \\
    --baseline-dir outputs/eddy_cloud_fair \\
    --baseline-ckpt outputs/eddy/best.pt \\
    --enh-dir outputs/eddy_enh7_cloud_fair \\
    --conf 0.25 \\
    --out submission/figures/eddy_3ch_vs_7ch_fig3-3.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FIG_FOOTER = (
    "底行两模型均在 val 集上以 conf≥{conf:.2f} 重绘 batch 拼图（RGB 底图 + Ultralytics 框/掩膜）。"
    "{enh_label} 在同阈值下候选框更少、输出更保守；3ch 召回更高，与表 5-4 中 3ch mAP 高于 7ch 的定量结论一致。"
)


def _read_map50(json_path: Path) -> float | None:
    if not json_path.is_file():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    v = (data.get("metrics") or {}).get("mask_map50")
    return float(v) if v is not None else None


def _rgb_from_png(png: Path) -> np.ndarray:
    return np.asarray(Image.open(png).convert("RGB"))


def _val_pngs(val_dir: Path) -> list[Path]:
    return sorted(val_dir.glob("*.png")) + sorted(val_dir.glob("*.jpg"))


def _tile_mosaic(
    images: list[np.ndarray],
    ncol: int = 4,
    *,
    ref_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if not images:
        raise ValueError("无图像可拼图")
    nrow = int(np.ceil(len(images) / ncol))
    if ref_shape is not None:
        canvas_h, canvas_w = ref_shape
        cell_h = max(1, canvas_h // nrow)
        cell_w = max(1, canvas_w // ncol)
    else:
        cell_h = min(im.shape[0] for im in images)
        cell_w = min(im.shape[1] for im in images)

    cells: list[np.ndarray] = []
    for im in images:
        arr = im if im.dtype == np.uint8 else (np.clip(im, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr).resize((cell_w, cell_h), Image.BILINEAR)
        cells.append(np.asarray(pil))
    pad = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 32
    while len(cells) < nrow * ncol:
        cells.append(pad)
    rows = [np.hstack(cells[r * ncol : (r + 1) * ncol]) for r in range(nrow)]
    return np.vstack(rows)


def _parse_val_png_stem(stem: str) -> tuple[str, int]:
    m = re.match(r"^(.+)_t(\d+)$", stem)
    if not m:
        raise ValueError(f"无法解析时间步: {stem}")
    return m.group(1), int(m.group(2))


def _pick_da(ds, names: tuple[str, ...]):
    lower = {str(k).lower(): k for k in ds.data_vars}
    for n in names:
        if n.lower() in lower:
            return ds[lower[n.lower()]]
    raise KeyError(f"未找到变量 {names}")


def _physics_stack_from_nc(nc_path: Path, time_idx: int, *, n_ch: int) -> np.ndarray:
    from src.eddy.stacked_physics import build_physics_stacked_hw7, build_physics_stacked_hw8
    from src.preprocess.eddy_physics import okubo_weiss_and_vorticity
    from src.utils.xarray_nc_open import open_xr_dataset_compat

    tmp = None
    ds = None
    try:
        ds, tmp = open_xr_dataset_compat(nc_path)
        adt = _pick_da(ds, ("adt", "ADT"))
        ug = _pick_da(ds, ("ugos", "UGOS"))
        vg = _pick_da(ds, ("vgos", "VGOS"))
        _sp = {"latitude", "longitude", "lat", "lon"}
        tdim = [d for d in adt.dims if d not in _sp][0]
        a = np.asarray(adt.isel({tdim: time_idx}).values, dtype=np.float64)
        u = np.asarray(ug.isel({tdim: time_idx}).values, dtype=np.float64)
        v = np.asarray(vg.isel({tdim: time_idx}).values, dtype=np.float64)
        lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
        zeta, ow = okubo_weiss_and_vorticity(u, v, lat, lon)
        if n_ch == 7:
            return build_physics_stacked_hw7(a, u, v, zeta, ow)
        return build_physics_stacked_hw8(a, u, v, zeta, ow)
    finally:
        if ds is not None:
            ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _pred_tile(
    model,
    val_png: Path,
    *,
    stack: np.ndarray | None,
    conf: float,
) -> tuple[np.ndarray, int]:
    import cv2

    bgr = cv2.cvtColor(_rgb_from_png(val_png), cv2.COLOR_RGB2BGR)
    if stack is not None:
        res = model.predict(stack.astype(np.float32), conf=conf, verbose=False)[0]
    else:
        res = model.predict(str(val_png), conf=conf, verbose=False)[0]
    n = int(len(res.boxes)) if res.boxes is not None else 0
    plotted = res.plot(img=bgr, conf=True, line_width=1)
    arr = np.asarray(plotted)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.05 else arr.astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), n


def _infer_physics_channels(enh_dir: Path) -> int:
    from src.utils.config import load_yaml, resolve_path

    p = resolve_path(f"data/processed/{enh_dir.name}/dataset.yaml")
    if p.is_file():
        ch = int(load_yaml(p).get("channels", 8))
        if ch in (7, 8):
            return ch
    if "enh7" in enh_dir.name.lower():
        return 7
    return 8


def _synth_val_pred_mosaic(
    ckpt: Path,
    val_dir: Path,
    nc_path: Path | None,
    *,
    conf: float,
    ref_shape: tuple[int, int] | None = None,
    ncol: int = 4,
    physics_channels: int = 8,
) -> tuple[np.ndarray | None, int]:
    from ultralytics import YOLO

    paths = _val_pngs(val_dir)
    if not paths or not ckpt.is_file():
        return None, 0
    model = YOLO(str(ckpt))
    tiles: list[np.ndarray] = []
    total = 0
    for png in paths:
        try:
            if nc_path is not None:
                stem, t_idx = _parse_val_png_stem(png.stem)
                if stem != nc_path.stem:
                    continue
                stack = _physics_stack_from_nc(nc_path, t_idx, n_ch=physics_channels)
                tile, n = _pred_tile(model, png, stack=stack, conf=conf)
            else:
                tile, n = _pred_tile(model, png, stack=None, conf=conf)
            tiles.append(tile)
            total += n
        except Exception:
            continue
    if not tiles:
        return None, 0
    return _tile_mosaic(tiles, ncol=ncol, ref_shape=ref_shape), total


def _panel_or_placeholder(ax, img: np.ndarray | None, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if img is not None:
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "（本地无该图）", ha="center", va="center", fontsize=11, color="#666")
        ax.set_facecolor("#f0f0f0")


def main() -> None:
    from src.utils.config import resolve_path

    ap = argparse.ArgumentParser(description="3ch vs 7ch/8ch 对比图（指标 + 统一 conf 预测）")
    ap.add_argument(
        "--baseline-dir",
        type=str,
        default="outputs/eddy_cloud_fair",
        help="3ch 指标目录（metrics_summary_*.json）",
    )
    ap.add_argument(
        "--baseline-ckpt",
        type=str,
        default="",
        help="3ch 权重；默认 {baseline-dir}/best.pt，不存在则回退 outputs/eddy/best.pt",
    )
    ap.add_argument(
        "--enh-dir",
        type=str,
        default="outputs/eddy_enh7_cloud_fair",
        help="7ch/8ch 指标与权重目录",
    )
    ap.add_argument(
        "--enh-ckpt",
        type=str,
        default="",
        help="7ch/8ch 权重；默认 {enh-dir}/best.pt",
    )
    ap.add_argument("--conf", type=float, default=0.25, help="底行两模型统一置信度阈值")
    ap.add_argument("--nc-val", type=str, default="", help="8ch 推理用 NC")
    ap.add_argument(
        "--out",
        type=str,
        default="submission/figures/eddy_3ch_vs_7ch_fig3-3.png",
    )
    args = ap.parse_args()

    base = resolve_path(args.baseline_dir)
    enh = resolve_path(args.enh_dir)
    val_dir = resolve_path("data/processed/eddy/images/val")
    nc_val = resolve_path(args.nc_val) if args.nc_val.strip() else resolve_path(
        "服创数据集/中尺度涡识别/20240101_20241231.nc"
    )

    def _resolve_ckpt(cli: str, primary: Path, fallback: str) -> Path:
        if cli.strip():
            return resolve_path(cli.strip())
        if (primary / "best.pt").is_file():
            return primary / "best.pt"
        return resolve_path(fallback)

    ckpt3 = _resolve_ckpt(args.baseline_ckpt, base, "outputs/eddy/best.pt")
    ckpt_enh = _resolve_ckpt(args.enh_ckpt, enh, str(enh / "best.pt"))
    conf = float(args.conf)
    n_phy = _infer_physics_channels(enh)
    enh_label = f"{n_phy}ch 物理堆叠"

    metrics = {
        "3ch_val": _read_map50(base / "metrics_summary_val.json"),
        "3ch_test": _read_map50(base / "metrics_summary_test.json"),
        "enh_val": _read_map50(enh / "metrics_summary_val.json"),
        "enh_test": _read_map50(enh / "metrics_summary_test.json"),
    }

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    img_pr3, n3 = _synth_val_pred_mosaic(ckpt3, val_dir, None, conf=conf)
    ref_shape = (img_pr3.shape[0], img_pr3.shape[1]) if img_pr3 is not None else None
    img_pr8, n8 = _synth_val_pred_mosaic(
        ckpt_enh,
        val_dir,
        nc_val,
        conf=conf,
        ref_shape=ref_shape,
        physics_channels=n_phy,
    )

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.28, wspace=0.08)

    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(2, dtype=float)
    w = 0.34
    v3 = [metrics["3ch_val"], metrics["3ch_test"]]
    v_enh = [metrics["enh_val"], metrics["enh_test"]]
    bars3 = ax_bar.bar(x - w / 2, [v if v is not None else 0 for v in v3], w, label="3ch RGB 基线", color="#5B8FD8")
    bars_enh = ax_bar.bar(x + w / 2, [v if v is not None else 0 for v in v_enh], w, label=enh_label, color="#D97B4A")
    ax_bar.set_ylabel("mask mAP@0.5", fontsize=11)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["验证集 val", "测试集 test"])
    ax_bar.set_ylim(0, 1.0)
    ax_bar.axhline(0.75, color="#888", linestyle="--", linewidth=1, label="项目通过线 0.75")
    ax_bar.legend(loc="upper left", fontsize=9)
    ax_bar.set_title("定量对比（Ultralytics 实例分割，同一划分与伪标签）", fontsize=11)
    for bar_group, vals in ((bars3, v3), (bars_enh, v_enh)):
        for bar, val in zip(bar_group, vals):
            if val is None:
                continue
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    if metrics["3ch_val"] is not None and metrics["enh_val"] is not None:
        dv = metrics["enh_val"] - metrics["3ch_val"]
        dt = (metrics["enh_test"] or 0) - (metrics["3ch_test"] or 0)
        ax_bar.text(
            0.98,
            0.95,
            f"Δ val = {dv:+.3f}  |  Δ test = {dt:+.3f}",
            transform=ax_bar.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.text(0.25, 0.92, "3ch RGB 基线", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.75, 0.92, enh_label, ha="center", fontsize=12, fontweight="bold")

    ax_p3 = fig.add_subplot(gs[1, 0])
    ax_p8 = fig.add_subplot(gs[1, 1])
    _panel_or_placeholder(
        ax_p3,
        img_pr3,
        f"3ch 基线 · val 预测（conf≥{conf:.2f}，共 {n3} 框）",
    )
    _panel_or_placeholder(
        ax_p8,
        img_pr8,
        f"{enh_label} · val 预测（conf≥{conf:.2f}，共 {n8} 框）",
    )

    fig.text(
        0.5,
        0.02,
        FIG_FOOTER.format(conf=conf, enh_label=enh_label),
        ha="center",
        fontsize=9,
        color="#333",
    )

    out_p = resolve_path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.88, bottom=0.08)
    fig.savefig(out_p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_p)
    print(f"val 框数 @ conf>={conf}: 3ch={n3}, {enh_label}={n8}")

    alt = resolve_path("outputs/eddy/figures/eddy_3ch_vs_8ch_input_compare.png")
    if alt.resolve() != out_p.resolve():
        alt.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(out_p, alt)
        print(alt)


if __name__ == "__main__":
    main()
