#!/usr/bin/env python3
"""
云端专项：水文预处理核查 + L0 vs 基线扩展指标对比。

对应文档：`docs/开发规划/下一步执行清单_云端评估前端与L0优化.md` §1、`docs/开发规划/后续开发工作清单_未完成项与云端L0专项.md` §2。

用法示例：
  python scripts/hydro_cloud_assessment.py audit --hydro-config config/hydro_hycom_l0.yaml --data-config config/data.yaml
  python scripts/hydro_cloud_assessment.py compare --split val \\
    --baseline-config config/hydro_hycom_l2.yaml --baseline-ckpt outputs/hydro_l2/best.pt \\
    --experiment-config config/experiments/hydro_hycom_l0_eos003.yaml \\
    --experiment-ckpt outputs/hydro_l0_eos003/best.pt \\
    --out-table-md submission/tables/hydro_l0_eos003_vs_l2_val.md \\
    --out-summary-json AutoDL/outputs/cloud/hydro_compare_val_summary_eos003.json
  # compare：z-space NRMSE + 物理空间 NRMSE（需 hydro_zscore.npz）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hydro.extended_metrics import evaluate_checkpoint_extended
from src.utils.config import load_yaml, pick_device, resolve_path


def _shape_report(path: Path, tag: str) -> dict[str, Any]:
    p = resolve_path(path)
    if not p.is_file():
        return {"tag": tag, "path": str(p), "exists": False}
    d = np.load(p)
    key = "X" if "X" in d.files else d.files[0]
    arr = d[key]
    return {
        "tag": tag,
        "path": str(p),
        "exists": True,
        "key": key,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_ratio": float(np.isfinite(arr).mean()) if arr.size else 0.0,
    }


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.hydro_config)
    data_cfg = load_yaml(args.data_config) if args.data_config else {}
    stats_dir = resolve_path(
        data_cfg.get("normalization", {}).get("stats_dir", "data/processed/stats")
    )
    stats_npz = stats_dir / "hydro_zscore.npz"
    meta_json = stats_dir / "hydro_preprocess_meta.json"

    paths = cfg["paths"]
    data = cfg["data"]
    tin = int(data["input_steps"])
    tout = int(data["output_steps"])
    gh, gw = int(data["grid_shape"][0]), int(data["grid_shape"][1])
    n_in = len(data["input_features"])
    n_out = len(data["target_features"])

    report: dict[str, Any] = {
        "hydro_config": str(resolve_path(args.hydro_config)),
        "expected": {
            "input_steps": tin,
            "output_steps": tout,
            "grid_shape": [gh, gw],
            "input_channels": n_in,
            "output_channels": n_out,
            "input_features": data["input_features"],
            "target_features": data["target_features"],
        },
        "issues": [],
        "splits": {},
        "stats": {},
        "meta_json": None,
    }

    for split in ("train", "val", "test"):
        xd = paths.get(f"{split}_data")
        yl = paths.get(f"{split}_label")
        if not xd or not yl:
            report["issues"].append(f"配置缺少 {split}_data 或 {split}_label")
            continue
        rx = _shape_report(Path(xd), f"{split}_X")
        ry = _shape_report(Path(yl), f"{split}_y")
        report["splits"][split] = {"X": rx, "y": ry}
        if not rx.get("exists") or not ry.get("exists"):
            report["issues"].append(f"{split}: npz 文件缺失")
            continue
        sx, sy = rx["shape"], ry["shape"]
        if sx[0] != sy[0]:
            report["issues"].append(f"{split}: 样本数不一致 X[0]={sx[0]} y[0]={sy[0]}")
        if len(sx) == 5 and len(sy) == 5:
            if sx[1] != tin:
                report["issues"].append(f"{split}: X 时间维={sx[1]} 期望 input_steps={tin}")
            if sy[1] != tout:
                report["issues"].append(f"{split}: y 时间维={sy[1]} 期望 output_steps={tout}")
            if sx[2] != gh or sx[3] != gw:
                report["issues"].append(f"{split}: X 空间={sx[2:4]} 期望 {[gh, gw]}")
            if sx[-1] != n_in or sy[-1] != n_out:
                report["issues"].append(
                    f"{split}: 通道 X_c={sx[-1]} y_c={sy[-1]} 期望 in={n_in} out={n_out}"
                )
        if rx.get("finite_ratio", 1.0) < 0.99 or ry.get("finite_ratio", 1.0) < 0.99:
            report["issues"].append(
                f"{split}: 非有限值比例偏高 X={rx.get('finite_ratio')} y={ry.get('finite_ratio')}"
            )

    if stats_npz.is_file():
        z = np.load(stats_npz)
        mean = np.asarray(z["mean"]).reshape(-1)
        std = np.asarray(z["std"]).reshape(-1)
        feats: list[str] = []
        if "features" in z.files:
            feats = [str(x) for x in z["features"].tolist()]
        std_min = float(np.nanmin(std)) if np.isfinite(std).any() else float("nan")
        report["stats"] = {
            "path": str(stats_npz),
            "mean_len": len(mean),
            "std_len": len(std),
            "std_min": std_min,
            "features_order": feats,
        }
        if feats:
            exp_in = list(data["input_features"])
            if feats != exp_in:
                if len(feats) < len(exp_in) and feats == exp_in[: len(feats)]:
                    report["issues"].append(
                        "hydro_zscore 仅含部分通道统计量，与当前 input_features 不全一致（需按当前配置重算 zscore）"
                    )
                else:
                    report["issues"].append(
                        f"hydro_zscore features 与 yaml input_features 不一致: {feats} vs {exp_in}"
                    )
        if len(mean) < n_in:
            report["issues"].append("hydro_zscore mean 长度不足以覆盖输入通道")
        elif np.isfinite(std).all() and float(np.nanmin(std)) < 1e-6:
            report["issues"].append("hydro_zscore 存在极小 std（量纲或统计失效风险）")
        elif bool(np.isnan(std).any()) or bool(np.isnan(mean).any()):
            report["issues"].append("hydro_zscore mean/std 含 NaN，请检查预处理")
    else:
        report["issues"].append(f"缺失统计文件: {stats_npz}")

    if meta_json.is_file():
        report["meta_json"] = str(meta_json)
        try:
            meta_full = json.loads(meta_json.read_text(encoding="utf-8"))
            report["meta_preview"] = {k: meta_full.get(k) for k in ("window_stride", "split_mode", "hydro_config") if k in meta_full}
        except json.JSONDecodeError as e:
            report["issues"].append(f"hydro_preprocess_meta.json 解析失败: {e}")

    report["overall_ok"] = len(report["issues"]) == 0

    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False, indent=2, default=str))
    if args.out_json:
        outp = resolve_path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {outp}")

    if not report["overall_ok"]:
        print("\n核查存在问题，见 issues 字段；建议先修数据再继续对比评估。")
        return 1
    print("\n核查通过（issues 为空）。")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    base_cfg = load_yaml(args.baseline_config)
    exp_cfg = load_yaml(args.experiment_config)

    def _chk_paths_match() -> list[str]:
        keys = ["val_data", "val_label", "test_data", "test_label"]
        bad: list[str] = []
        for k in keys:
            vb = base_cfg["paths"].get(k)
            ve = exp_cfg["paths"].get(k)
            if vb != ve:
                bad.append(f"{k}: baseline={vb} experiment={ve}")
        return bad

    mismatches = _chk_paths_match()
    if mismatches:
        print("实验与基线 data npz 路径不一致（对比将失去意义）：\n" + "\n".join(mismatches))
        if not args.force:
            print("若仍要运行，请加 --force")
            return 3

    device = torch.device(pick_device(exp_cfg["train"].get("device", "cpu")))

    stats_path: Path | None = None
    if args.stats_npz:
        stats_path = resolve_path(args.stats_npz)
    else:
        default_stats = resolve_path("data/processed/stats/hydro_zscore.npz")
        if default_stats.is_file():
            stats_path = default_stats

    ckpt_b = resolve_path(args.baseline_ckpt)
    ckpt_e = resolve_path(args.experiment_ckpt)
    if not ckpt_b.is_file():
        print(f"基线权重不存在: {ckpt_b}")
        return 2
    if not ckpt_e.is_file():
        print(f"实验权重不存在: {ckpt_e}")
        return 2

    m_base = evaluate_checkpoint_extended(
        base_cfg,
        ckpt_b,
        device,
        split=args.split,
        stats_npz_path=stats_path,
        max_batches=args.max_batches,
    )
    m_exp = evaluate_checkpoint_extended(
        exp_cfg,
        ckpt_e,
        device,
        split=args.split,
        stats_npz_path=stats_path,
        max_batches=args.max_batches,
    )

    has_phys = "nrmse_physical_per_feature" in m_base and "nrmse_physical_per_feature" in m_exp
    if not has_phys:
        if stats_path is None or not stats_path.is_file():
            print(
                "提示: 未提供可用的 --stats-npz，跳过物理空间 NRMSE；"
                "请指定 data/processed/stats/hydro_zscore.npz"
            )
        else:
            print("提示: stats 已加载但未产出 nrmse_physical_per_feature，请检查 mean/std 通道数")

    feats = list(base_cfg["data"]["target_features"])
    rows: list[dict[str, Any]] = []
    for name in feats:
        sk_b = m_base["skill_vs_persistence"].get(name)
        sk_e = m_exp["skill_vs_persistence"].get(name)
        pr_b = m_base["pearson_per_feature"].get(name)
        pr_e = m_exp["pearson_per_feature"].get(name)
        row: dict[str, Any] = {
            "feature": name,
            "baseline_mae": float(m_base["mae_per_feature"][name]),
            "experiment_mae": float(m_exp["mae_per_feature"][name]),
            "baseline_rmse_norm": float(m_base["rmse_per_feature"][name]),
            "experiment_rmse_norm": float(m_exp["rmse_per_feature"][name]),
            "baseline_nrmse": float(m_base["nrmse_per_feature"][name]),
            "experiment_nrmse": float(m_exp["nrmse_per_feature"][name]),
            "baseline_skill": sk_b if sk_b is not None else "",
            "experiment_skill": sk_e if sk_e is not None else "",
            "baseline_pearson": pr_b if pr_b is not None else "",
            "experiment_pearson": pr_e if pr_e is not None else "",
        }
        rpb = (
            m_base.get("rmse_physical_scale", {}).get(name) if m_base.get("rmse_physical_scale") else None
        )
        rpe = (
            m_exp.get("rmse_physical_scale", {}).get(name) if m_exp.get("rmse_physical_scale") else None
        )
        if rpb is not None and rpe is not None:
            row["baseline_rmse_phys"] = float(rpb)
            row["experiment_rmse_phys"] = float(rpe)
        if has_phys:
            row["baseline_nrmse_phys"] = float(m_base["nrmse_physical_per_feature"][name])
            row["experiment_nrmse_phys"] = float(m_exp["nrmse_physical_per_feature"][name])
            row["baseline_rmse_phys_denorm"] = float(m_base["rmse_physical_per_feature"][name])
            row["experiment_rmse_phys_denorm"] = float(m_exp["rmse_physical_per_feature"][name])
        rows.append(row)

    base_skill_avg = m_base.get("skill_avg")
    exp_skill_avg = m_exp.get("skill_avg")

    verdict_mae = float(m_exp["mae_avg"]) < float(m_base["mae_avg"])
    verdict_nrmse = float(m_exp["nrmse_avg"]) < float(m_base["nrmse_avg"])
    verdict_nrmse_phys = None
    if has_phys:
        verdict_nrmse_phys = float(m_exp["nrmse_physical_avg"]) < float(m_base["nrmse_physical_avg"])
    verdict_skill = False
    if base_skill_avg is not None and exp_skill_avg is not None:
        verdict_skill = float(exp_skill_avg) > float(base_skill_avg)

    summary_block = {
        "split": args.split,
        "baseline_ckpt": str(ckpt_b),
        "experiment_ckpt": str(ckpt_e),
        "baseline_mae_avg": float(m_base["mae_avg"]),
        "experiment_mae_avg": float(m_exp["mae_avg"]),
        "baseline_rmse_avg_norm": float(m_base["rmse_avg"]),
        "experiment_rmse_avg_norm": float(m_exp["rmse_avg"]),
        "baseline_nrmse_avg": float(m_base["nrmse_avg"]),
        "experiment_nrmse_avg": float(m_exp["nrmse_avg"]),
        "baseline_skill_avg": base_skill_avg,
        "experiment_skill_avg": exp_skill_avg,
        "baseline_pearson_avg": m_base.get("pearson_avg"),
        "experiment_pearson_avg": m_exp.get("pearson_avg"),
        "conclusion_mae_avg_experiment_lower": verdict_mae,
        "conclusion_nrmse_avg_experiment_lower": verdict_nrmse,
        "conclusion_skill_avg_experiment_higher": verdict_skill,
        "material_line_l0_stable_improve": verdict_mae and verdict_skill if base_skill_avg is not None else None,
        "definitions": {
            "nrmse": "与 src/hydro/eval.py 一致：RMSE / mean(|y|)，在 z-score 目标上按 B×T×H×W 聚合",
            "nrmse_physical": "反标准化后 RMSE_phys / mean(|y_phys|)，stats 来自 hydro_zscore.npz",
            "skill_vs_persistence": "1 - MSE_model / MSE_naive ，naive = 复制输入末时刻到整个预报窗",
            "rmse_physical_scale": "RMSE(z)×std，与 rmse_physical_per_feature 对误差等价，NRMSE 须用 nrmse_physical",
        },
    }
    if has_phys:
        summary_block["stats_npz"] = str(stats_path)
        summary_block["baseline_nrmse_physical_avg"] = float(m_base["nrmse_physical_avg"])
        summary_block["experiment_nrmse_physical_avg"] = float(m_exp["nrmse_physical_avg"])
        summary_block["baseline_rmse_physical_avg"] = float(m_base["rmse_physical_avg"])
        summary_block["experiment_rmse_physical_avg"] = float(m_exp["rmse_physical_avg"])
        summary_block["conclusion_nrmse_physical_avg_experiment_lower"] = verdict_nrmse_phys

    print(json.dumps({"summary": summary_block, "per_feature": rows}, ensure_ascii=False, indent=2, default=str))

    if args.out_table_md:
        md_path = resolve_path(args.out_table_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines_md = [
            "# 水文：实验模型 vs 基线（扩展指标）",
            "",
            f"- split: **{args.split}**",
            f"- baseline: `{ckpt_b}` ",
            f"- experiment: `{ckpt_e}` ",
            "",
            "## 汇总（材料结论占位）",
            "",
            "| 项 | baseline | experiment | 结论（实验优于基线？） |",
            "| --- | --- | --- | --- |",
            f"| MAE_avg (norm空间) | {m_base['mae_avg']:.6g} | {m_exp['mae_avg']:.6g} | {'是' if verdict_mae else '否'} |",
            f"| NRMSE_avg（z-score 空间） | {m_base['nrmse_avg']:.6g} | {m_exp['nrmse_avg']:.6g} | {'是' if verdict_nrmse else '否'} |",
        ]
        if has_phys:
            lines_md.append(
                f"| NRMSE_avg（物理空间） | {m_base['nrmse_physical_avg']:.6g} | "
                f"{m_exp['nrmse_physical_avg']:.6g} | {'是' if verdict_nrmse_phys else '否'} |"
            )
        if base_skill_avg is not None:
            yes_skill = '是' if verdict_skill else '否'
            lines_md.append(
                f"| Skill_avg (vs persistence) | {float(base_skill_avg):.6g} | "
                f"{float(exp_skill_avg or 0):.6g} | {yes_skill} |"
            )
        if has_phys:
            lines_md += [
                "",
                "## 通道明细（z-score 与物理 NRMSE）",
                "",
                "| channel | NRMSE_B(z) | NRMSE_E(z) | NRMSE_B(phys) | NRMSE_E(phys) | RMSE_B(phys) | RMSE_E(phys) | Skill_B | Skill_E |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for r in rows:
                lines_md.append(
                    f"| {r['feature']} | {r['baseline_nrmse']:.6g} | {r['experiment_nrmse']:.6g} | "
                    f"{r['baseline_nrmse_phys']:.6g} | {r['experiment_nrmse_phys']:.6g} | "
                    f"{r['baseline_rmse_phys_denorm']:.6g} | {r['experiment_rmse_phys_denorm']:.6g} | "
                    f"{r['baseline_skill']} | {r['experiment_skill']} |"
                )
        else:
            lines_md += [
                "",
                "## 通道明细",
                "",
                "| channel | MAE_B | MAE_E | RMSE_B | RMSE_E | NRMSE_B | NRMSE_E | Skill_B | Skill_E | r_B | r_E |",
            ]
            lines_md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in rows:
                lines_md.append(
                    f"| {r['feature']} | {r['baseline_mae']:.6g} | {r['experiment_mae']:.6g} | "
                    f"{r['baseline_rmse_norm']:.6g} | {r['experiment_rmse_norm']:.6g} | "
                    f"{r['baseline_nrmse']:.6g} | {r['experiment_nrmse']:.6g} | "
                    f"{r['baseline_skill']} | {r['experiment_skill']} | {r['baseline_pearson']} | {r['experiment_pearson']} |"
                )
        md_path.write_text("\n".join(lines_md), encoding="utf-8")
        print(f"wrote {md_path}")

    if args.out_csv:
        csv_path = resolve_path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {csv_path}")

    if args.out_summary_json:
        sj = resolve_path(args.out_summary_json)
        sj.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary_block, "per_feature": rows, "raw_baseline_metrics": m_base, "raw_experiment_metrics": m_exp}
        with sj.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"wrote {sj}")

    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="云端：水文预处理核查与 L0/基线扩展对比")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="核查 npz 形状、通道、统计量与 meta")
    p_audit.add_argument("--hydro-config", type=str, default="config/hydro_hycom_l0.yaml")
    p_audit.add_argument("--data-config", type=str, default="config/data.yaml")
    p_audit.add_argument("--out-json", type=str, default="", help="可选写出完整报告 JSON")
    p_audit.set_defaults(func=cmd_audit)

    p_cmp = sub.add_parser("compare", help="同一 val/test 上对比基线与实验 checkpoint")
    p_cmp.add_argument("--split", type=str, choices=("val", "test"), default="val")
    p_cmp.add_argument("--baseline-config", type=str, default="config/hydro_hycom_l2.yaml")
    p_cmp.add_argument("--baseline-ckpt", type=str, default="outputs/hydro_l2/best.pt")
    p_cmp.add_argument("--experiment-config", type=str, default="config/hydro_hycom_l0.yaml")
    p_cmp.add_argument("--experiment-ckpt", type=str, default="outputs/hydro_l0/best.pt")
    p_cmp.add_argument(
        "--stats-npz",
        type=str,
        default="",
        help="Z-score 统计量；默认尝试 data/processed/stats/hydro_zscore.npz，用于物理 RMSE/NRMSE",
    )
    p_cmp.add_argument("--max-batches", type=int, default=None, help="仅调试用，限制批次数")
    p_cmp.add_argument("--force", action="store_true", help="data 路径不一致时仍运行")
    p_cmp.add_argument("--out-table-md", type=str, default="")
    p_cmp.add_argument("--out-csv", type=str, default="")
    p_cmp.add_argument("--out-summary-json", type=str, default="")
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
