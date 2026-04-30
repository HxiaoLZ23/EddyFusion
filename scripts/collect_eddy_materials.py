from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_if_exists(src: Path, dst: Path, copied: list[Path]) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="收集 eddy 训练/评估图用于企业材料②")
    parser.add_argument("--src-root", type=str, default="AutoDL/outputs/eddy")
    parser.add_argument("--out-dir", type=str, default="submission/figures/eddy_latest")
    args = parser.parse_args()

    src = resolve_path(args.src_root)
    out = ensure_dir(args.out_dir)
    copied: list[Path] = []

    # 优先收集最有展示价值的图
    picks = [
        ("train/results.png", "results.png"),
        ("train/confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
        ("train/MaskPR_curve.png", "mask_pr_curve_train.png"),
        ("train/MaskF1_curve.png", "mask_f1_curve_train.png"),
        ("train/val_batch0_pred.jpg", "val_batch0_pred_train.jpg"),
        ("train/val_batch0_labels.jpg", "val_batch0_labels_train.jpg"),
        ("eval_val/confusion_matrix_normalized.png", "confusion_matrix_normalized_eval.jpg"),
        ("eval_val/MaskPR_curve.png", "mask_pr_curve_eval.jpg"),
        ("eval_val/MaskF1_curve.png", "mask_f1_curve_eval.jpg"),
        ("eval_val/val_batch0_pred.jpg", "val_batch0_pred_eval.jpg"),
        ("eval_val/val_batch0_labels.jpg", "val_batch0_labels_eval.jpg"),
        ("metrics_summary_val.json", "metrics_summary_val.json"),
        ("best.pt", "best.pt"),
    ]
    for rel_src, rel_dst in picks:
        _copy_if_exists(src / rel_src, out / rel_dst, copied)

    index = out / "README.md"
    lines = [
        "# Eddy 材料收集清单",
        "",
        f"- 来源目录：`{src}`",
        f"- 目标目录：`{out}`",
        f"- 已复制文件数：**{len(copied)}**",
        "",
        "## 文件列表",
    ]
    for p in copied:
        lines.append(f"- `{p.name}`")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"copied={len(copied)} -> {out}")
    print(f"index={index}")


if __name__ == "__main__":
    main()
