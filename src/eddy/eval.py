"""涡旋评估：YOLOv8-seg ``model.val``，按 split 写出 ``metrics_summary_<split>.json``（字段符合项目约定）。"""

from __future__ import annotations

import argparse

from src.utils.config import load_yaml, resolve_path
from src.utils.metrics import write_metrics_json


def _pick_map50(metrics) -> float:
    seg = getattr(metrics, "seg", None)
    if seg is not None and hasattr(seg, "map50"):
        return float(seg.map50)
    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "map50"):
        return float(box.map50)
    rd = getattr(metrics, "results_dict", None)
    if isinstance(rd, dict):
        for k in (
            "metrics/mAP50(M)",
            "metrics/mAP50(B)",
            "metrics/mAP50",
            "mAP50",
        ):
            if k in rd and rd[k] is not None:
                return float(rd[k])
    return 0.0


def _run_one_split(
    *,
    model,
    dataset_yaml,
    split: str,
    out_dir,
    level: int,
    metrics_stem: str,
) -> tuple[float, Path]:
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        project=str(out_dir),
        name=f"eval_{split}",
        exist_ok=True,
    )
    map50 = _pick_map50(metrics)
    passed = map50 >= 0.75
    payload_metrics = {
        "mask_map50": map50,
        "split": split,
        "note": "Ultralytics seg mAP@0.5；与命题方 IoU 口径需人工核对",
    }

    mf = metrics_stem
    mp = resolve_path(mf)
    outp = mp.parent / f"{mp.stem}_{split}{mp.suffix}"
    write_metrics_json(
        outp,
        module="eddy",
        level=level,
        metrics=payload_metrics,
        passed=passed,
        tags={"level": level, "eval_split": split},
    )
    return map50, outp


def main() -> None:
    parser = argparse.ArgumentParser(description="涡旋分割 mAP 评估（固定 dataset.yaml 划分）")
    parser.add_argument("--config", type=str, default="config/eddy.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/eddy/best.pt")
    parser.add_argument(
        "--splits",
        type=str,
        default="val",
        help="逗号分隔：val / test / val,test；需 dataset.yaml 含对应 images 划分",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        out = resolve_path(cfg["paths"]["output_dir"])
        alt = out / "last.pt"
        raise FileNotFoundError(f"未找到权重: {ckpt}" + (f"\n可试: --ckpt {alt}" if alt.is_file() else ""))

    dataset_yaml = resolve_path(cfg["paths"]["dataset_yaml"])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"未找到数据集 yaml: {dataset_yaml}，无法执行 val（与训练使用同一划分文件）"
        )

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    allowed = {"val", "test"}
    bad = set(splits) - allowed
    if bad:
        raise ValueError(f"不支持的 split: {bad}（仅允许 val 与 test）")

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    level = int(cfg["meta"]["level"])
    mf = cfg.get("eval", {}).get("metrics_file", "outputs/eddy/metrics_summary.json")

    for sp in splits:
        map50, outp = _run_one_split(
            model=model,
            dataset_yaml=dataset_yaml,
            split=sp,
            out_dir=out_dir,
            level=level,
            metrics_stem=mf,
        )
        print(f"wrote {outp}")
        print(f"[{sp}] mask_map50={map50:.6f}")


if __name__ == "__main__":
    main()
