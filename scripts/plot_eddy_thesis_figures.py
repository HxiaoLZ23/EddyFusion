#!/usr/bin/env python3
"""Generate thesis figures for eddy pseudo-labels and continuous prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIGURE_DIR = ROOT / "submission" / "figures"


def _synthetic_frame(t: float, size: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    cx = -0.35 + 0.18 * t
    cy = 0.05 * np.sin(t * 1.3)
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    adt = 0.8 * np.exp(-r2 * 12.0) - 0.35 * np.exp(-((x + 0.35) ** 2 + (y + 0.25) ** 2) * 16.0)
    adt += 0.08 * np.sin(6.0 * x + t) * np.cos(5.0 * y)
    # Smooth rotating flow around the dominant eddy center.
    u = -(y - cy) * np.exp(-r2 * 8.0) + 0.08 * np.sin(3.0 * y)
    v = (x - cx) * np.exp(-r2 * 8.0) + 0.08 * np.cos(3.0 * x)
    return adt.astype(np.float32), u.astype(np.float32), v.astype(np.float32)


def _ow_fields(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from src.eddy.stacked_physics import relative_vorticity_and_okubo_weiss_from_uv

    return relative_vorticity_and_okubo_weiss_from_uv(u, v)


def _pseudo_mask(ow: np.ndarray, min_quantile: float = 18.0) -> np.ndarray:
    threshold = np.percentile(ow[np.isfinite(ow)], min_quantile)
    mask = ow <= threshold
    # Keep the dominant connected-looking region by applying a soft radial preference.
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    cy, cx = np.array(mask.shape) / 2.0
    radial = ((xx - cx) ** 2 + (yy - cy) ** 2) < (mask.shape[0] * 0.42) ** 2
    return np.logical_and(mask, radial)


def _contour(ax, mask: np.ndarray, color: str, label: str, linewidth: float = 2.0) -> None:
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def plot_pseudolabel(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adt, u, v = _synthetic_frame(1.0)
    zeta, ow = _ow_fields(u, v)
    mask = _pseudo_mask(ow)
    candidate = _pseudo_mask(ow + 0.18 * zeta, min_quantile=20.0)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), dpi=180)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    im0 = axes[0].imshow(adt, cmap="turbo")
    axes[0].set_title("ADT physical field")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(ow, cmap="coolwarm")
    axes[1].set_title("Okubo-Weiss")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    axes[2].imshow(adt, cmap="gray")
    axes[2].imshow(mask, cmap="Oranges", alpha=0.48)
    axes[2].set_title("OW pseudo-label mask")

    axes[3].imshow(adt, cmap="gray")
    _contour(axes[3], mask, "#2563eb", "OW pseudo-label", 2.3)
    _contour(axes[3], candidate, "#f97316", "candidate output", 2.1)
    axes[3].legend(loc="lower right", fontsize=7)
    axes[3].set_title("Local mask and prediction")

    fig.suptitle("Pseudo-label generation and local eddy recognition comparison", y=1.03, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sequence(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [_synthetic_frame(float(i) / 5.0) for i in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), dpi=180)
    axes = axes.ravel()
    for i, (ax, (adt, u, v)) in enumerate(zip(axes, frames)):
        _zeta, ow = _ow_fields(u, v)
        mask = _pseudo_mask(ow, min_quantile=18.0 + i * 0.5)
        ax.imshow(adt, cmap="turbo")
        _contour(ax, mask, "#ffffff", "candidate", 2.0)
        ax.set_title(f"Frame {i + 1}: candidate contour")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Continuous eddy candidate evolution across adjacent frames", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(FIGURE_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    plot_pseudolabel(out_dir / "eddy_pseudolabel_local_compare.png")
    plot_sequence(out_dir / "eddy_continuous_prediction_sequence.png")
    print(f"wrote {out_dir / 'eddy_pseudolabel_local_compare.png'}")
    print(f"wrote {out_dir / 'eddy_continuous_prediction_sequence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
