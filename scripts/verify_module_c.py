#!/usr/bin/env python3
"""模块 C 自检：旧权重兼容(B)、异常分级(C)、DTW 重排(D)。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.detect import compute_anomaly_assessment, rerank_candidates_by_dtw
from src.anomaly.model import build_model
from src.utils.config import load_yaml


def verify_old_head_compat(hidden_dim: int = 128) -> None:
    """B: 用合成 old head.weight/head.bias 验证 load_state_dict 兼容。"""
    cfg = {"model": {"hidden_dim": hidden_dim, "num_layers": 1}}
    m = build_model(cfg)
    full = m.state_dict()
    old_sd: dict[str, torch.Tensor] = {}
    for k, v in full.items():
        if k.startswith("lstm."):
            old_sd[k] = v.clone()
    old_sd["head.weight"] = torch.cat([full["wind_head.weight"], full["wave_head.weight"]], dim=0).clone()
    old_sd["head.bias"] = torch.cat([full["wind_head.bias"], full["wave_head.bias"]], dim=0).clone()

    m2 = build_model(cfg)
    m2.load_state_dict(old_sd, strict=True)
    x = torch.randn(4, 12, 2)
    torch.testing.assert_close(m(x), m2(x), rtol=1e-5, atol=1e-5)
    print("[B] PASS: old head.* 权重可被双头模型加载，前向与张量与原模型一致。")


def verify_anomaly_levels() -> None:
    """C: 无信号 -> unknown；有观测-预测残差 -> 数值等级。"""
    ctx = {
        "start_time": "2020-01-01 00:00:00",
        "end_time": "2020-01-02 00:00:00",
        "lon_min": 118.0,
        "lon_max": 126.0,
        "lat_min": 31.0,
        "lat_max": 41.0,
    }

    out_no = compute_anomaly_assessment(ctx)
    assert out_no["anomaly_level"] == "unknown", out_no
    assert out_no.get("assessment_note"), out_no
    assert out_no.get("anomaly_index") is None
    print(f"[C.1] PASS: 无残差 -> level={out_no['anomaly_level']}, note 存在。")

    out_low = compute_anomaly_assessment(
        {
            **ctx,
            "wind_observed": 0.5,
            "wind_predicted": 0.0,
            "wave_observed": 0.5,
            "wave_predicted": 0.0,
            "wind_mean": 0.0,
            "wind_std": 1.0,
            "wave_mean": 0.0,
            "wave_std": 1.0,
        }
    )
    assert out_low["anomaly_level"] == "low" and isinstance(out_low["anomaly_index"], float)
    print(f"[C.2] PASS: 小残差 -> {out_low['anomaly_level']}, index={out_low['anomaly_index']:.4f}")

    out_high = compute_anomaly_assessment(
        {
            **ctx,
            "wind_residual": 6.0,
            "wave_residual": 6.0,
            "wind_mean": 0.0,
            "wind_std": 1.0,
            "wave_mean": 0.0,
            "wave_std": 1.0,
        }
    )
    assert out_high["anomaly_level"] == "high" and out_high["anomaly_index"] is not None
    print(f"[C.3] PASS: 大残差(z≈6) -> {out_high['anomaly_level']}, index={out_high['anomaly_index']:.4f}")


def verify_dtw_rerank() -> None:
    """D: 有 current_curve 时按 dtw 排序；无时切片原序 + reason。"""
    candidates = [
        {"event_id": "far_first", "sequence": [50.0, 50.0, 50.0]},
        {"event_id": "close_second", "sequence": [0.0, 0.0, 0.0]},
        {"event_id": "mid_third", "sequence": [2.0, 2.0, 2.0]},
    ]
    curve = [0.0, 0.0, 0.0]
    reranked, meta_on = rerank_candidates_by_dtw(candidates=candidates, current_curve=curve, top_k=3)
    assert meta_on["enabled"] is True, meta_on
    ids_on = [c["event_id"] for c in reranked]
    assert ids_on[0] == "close_second", ids_on
    assert "dtw_distance" in reranked[0]
    print(f"[D.1] PASS: 有曲线 -> 顺序 {ids_on}, dtw enabled。")

    slice_only, meta_off = rerank_candidates_by_dtw(candidates=candidates, current_curve=None, top_k=3)
    assert meta_off == {"enabled": False, "reason": "missing_current_curve"}, meta_off
    ids_off = [c["event_id"] for c in slice_only]
    assert ids_off == ["far_first", "close_second", "mid_third"], ids_off
    assert "dtw_distance" not in slice_only[0]
    print(f"[D.2] PASS: 无曲线 -> 原序切片 {ids_off}, meta={meta_off}")


def write_old_format_ckpt_for_eval(config_path: str, out_path: Path) -> None:
    """生成与旧 train 保存格式兼容的 checkpoint，供 eval.py 实测。"""
    cfg = load_yaml(config_path)
    m = build_model(cfg)
    full = m.state_dict()
    old_sd: dict[str, torch.Tensor] = {}
    for k, v in full.items():
        if k.startswith("lstm."):
            old_sd[k] = v.clone()
    old_sd["head.weight"] = torch.cat([full["wind_head.weight"], full["wave_head.weight"]], dim=0).clone()
    old_sd["head.bias"] = torch.cat([full["wind_head.bias"], full["wave_head.bias"]], dim=0).clone()

    payload = {"model": old_sd, "cfg": cfg, "epoch": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[B-extra] 已写入旧格式权重: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="模块 C B/C/D 自检")
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument(
        "--write-old-ckpt",
        type=str,
        default="",
        help="可选：写出旧 head 格式 ckpt 路径（用于跑 eval.py）",
    )
    parser.add_argument(
        "--run-eval-on-old",
        action="store_true",
        help="写入临时旧格式 ckpt 并调用 eval.py（需数据与权重路径可用）",
    )
    parser.add_argument("--eval-split", type=str, choices=("val", "test"), default="val")
    args = parser.parse_args()

    verify_old_head_compat()
    verify_anomaly_levels()
    verify_dtw_rerank()

    if args.write_old_ckpt:
        write_old_format_ckpt_for_eval(args.config, Path(args.write_old_ckpt))

    if args.run_eval_on_old:
        with NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            tmp = Path(tf.name)
        try:
            write_old_format_ckpt_for_eval(args.config, tmp)
            cmd = [
                sys.executable,
                str(REPO_ROOT / "src/anomaly/eval.py"),
                "--config",
                args.config,
                "--ckpt",
                str(tmp),
                "--split",
                args.eval_split,
            ]
            print("[B-extra] 运行:", " ".join(cmd))
            r = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if r.returncode != 0:
                raise SystemExit(r.returncode)
            print("[B-extra] PASS: eval.py 在旧格式 ckpt 上完成。")
        finally:
            tmp.unlink(missing_ok=True)

    print("\n全部 B/C/D 断言通过。")


if __name__ == "__main__":
    main()
