"""涡旋评估：YOLOv8-seg val，输出 metrics_summary_test.json（字段符合项目约定）。"""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/eddy.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/eddy/best.pt")
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
            f"未找到数据集 yaml: {dataset_yaml}，无法执行 val（与训练使用同一 data/processed/eddy/dataset.yaml）"
        )

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        project=str(out_dir),
        name="eval_val",
        exist_ok=True,
    )
    map50 = _pick_map50(metrics)
    level = int(cfg["meta"]["level"])
    # 赛题涡旋「准确率」指 IoU；基线用验证集 mask mAP50 作可汇总标量，命题方若另有脚本再替换
    passed = map50 >= 0.75
    payload_metrics = {
        "mask_map50": map50,
        "split": "val",
        "note": "Ultralytics seg mAP@0.5；与命题方 IoU 口径需人工核对",
    }

    mf = cfg.get("eval", {}).get("metrics_file", "outputs/eddy/metrics_summary.json")
    mp = resolve_path(mf)
    out_json = mp.parent / f"{mp.stem}_val{mp.suffix}"
    write_metrics_json(
        out_json,
        module="eddy",
        level=level,
        metrics=payload_metrics,
        passed=passed,
        tags={"level": level, "eval_split": "val"},
    )
    print(f"wrote {out_json}")
    print("metrics:", payload_metrics)


if __name__ == "__main__":
    main()
