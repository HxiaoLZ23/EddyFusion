"""涡旋评估：YOLOv8-seg ``model.val``，按 split 写出 ``metrics_summary_<split>.json``（字段符合项目约定）。"""

from __future__ import annotations

import argparse
import statistics
from typing import Any

from src.utils.config import load_yaml, resolve_path
from src.utils.metrics import write_metrics_json


def _float_attr(obj: Any, name: str) -> float | None:
    if obj is None or not hasattr(obj, name):
        return None
    try:
        v = getattr(obj, name)
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean_f1(seg: Any) -> float | None:
    f1 = getattr(seg, "f1", None)
    if f1 is None:
        return None
    try:
        seq = list(f1)
        if not seq:
            return None
        return float(statistics.mean(float(x) for x in seq))
    except (TypeError, ValueError, statistics.StatisticsError):
        return None


def _results_dict_float(rd: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        if k in rd and rd[k] is not None:
            try:
                return float(rd[k])
            except (TypeError, ValueError):
                continue
    return None


def _pick_map50(metrics: Any) -> float:
    seg = getattr(metrics, "seg", None)
    if seg is not None and hasattr(seg, "map50"):
        return float(seg.map50)
    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "map50"):
        return float(box.map50)
    rd = getattr(metrics, "results_dict", None)
    if isinstance(rd, dict):
        v = _results_dict_float(
            rd,
            "metrics/mAP50(M)",
            "metrics/mAP50(B)",
            "metrics/mAP50",
            "mAP50",
        )
        if v is not None:
            return v
    return 0.0


def _mask_metric_bundle(metrics: Any) -> dict[str, float]:
    """Ultralytics SegmentMetrics：mask 分支的 mAP50–95、P/R/F1 等（与 ``metrics.seg`` / ``results_dict`` 对齐）。"""
    out: dict[str, float] = {}
    seg = getattr(metrics, "seg", None)
    if seg is not None:
        mapping = (
            ("mask_map50_95", "map"),
            ("mask_map75", "map75"),
            ("mask_mean_precision", "mp"),
            ("mask_mean_recall", "mr"),
        )
        for json_key, attr in mapping:
            v = _float_attr(seg, attr)
            if v is not None:
                out[json_key] = v
        mf1 = _mean_f1(seg)
        if mf1 is not None:
            out["mask_mean_f1"] = mf1

    rd = getattr(metrics, "results_dict", None)
    if isinstance(rd, dict):
        fallbacks: list[tuple[str, tuple[str, ...]]] = [
            ("mask_map50_95", ("metrics/mAP50-95(M)", "metrics/mAP50-95")),
            ("mask_map75", ("metrics/mAP75(M)", "metrics/mAP75")),
            ("mask_mean_precision", ("metrics/precision(M)", "metrics/precision")),
            ("mask_mean_recall", ("metrics/recall(M)", "metrics/recall")),
        ]
        for json_key, keys in fallbacks:
            if json_key not in out:
                v = _results_dict_float(rd, *keys)
                if v is not None:
                    out[json_key] = v

    return out


def _run_one_split(
    *,
    model,
    dataset_yaml,
    split: str,
    out_dir,
    level: int,
    metrics_stem: str,
) -> tuple[float, Path, dict[str, Any]]:
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        project=str(out_dir),
        name=f"eval_{split}",
        exist_ok=True,
    )
    map50 = _pick_map50(metrics)
    passed = map50 >= 0.75
    extra = _mask_metric_bundle(metrics)
    payload_metrics: dict[str, Any] = {
        "mask_map50": map50,
        "split": split,
        "note": (
            "Ultralytics YOLO-seg：mask mAP@0.5 为 mask_map50；"
            "mask_map50_95 为 mAP@0.5:0.95；"
            "mask_mean_precision / mask_mean_recall / mask_mean_f1 为各类别均值（与 validator 默认策略一致）。"
            "与命题方 IoU 口径需人工核对。"
        ),
    }
    payload_metrics.update(extra)

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
    return map50, outp, payload_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="涡旋分割 mAP 评估（固定 dataset.yaml 划分）")
    parser.add_argument("--config", type=str, default="config/eddy.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/eddy_v6_b0_fair/best.pt")
    parser.add_argument(
        "--splits",
        type=str,
        default="val",
        help="逗号分隔：val / test / val,test；需 dataset.yaml 含对应 images 划分",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = resolve_path(cfg["paths"]["output_dir"])
    ckpt_arg = str(args.ckpt).replace("\\", "/")
    _config_default_ckpts = frozenset(
        {
            "outputs/eddy/best.pt",
            "outputs/eddy_v6_b0_fair/best.pt",
        }
    )
    if ckpt_arg in _config_default_ckpts or not resolve_path(args.ckpt).is_file():
        for candidate in (out / "best.pt", out / "last.pt"):
            if candidate.is_file():
                ckpt = candidate
                break
        else:
            raise FileNotFoundError(f"未找到权重: {out / 'best.pt'} 或 {out / 'last.pt'}")
    else:
        ckpt = resolve_path(args.ckpt)

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
        map50, outp, pay = _run_one_split(
            model=model,
            dataset_yaml=dataset_yaml,
            split=sp,
            out_dir=out_dir,
            level=level,
            metrics_stem=mf,
        )
        print(f"wrote {outp}")
        tail = ""
        m95 = pay.get("mask_map50_95")
        mpv = pay.get("mask_mean_precision")
        mrv = pay.get("mask_mean_recall")
        if m95 is not None and mpv is not None and mrv is not None:
            tail = f" mask_map50_95={float(m95):.6f} P={float(mpv):.6f} R={float(mrv):.6f}"
        print(f"[{sp}] mask_map50={map50:.6f}{tail}")


if __name__ == "__main__":
    main()
