#!/usr/bin/env python3
"""3ch 增通道消融：读结果 → 动态规划下一实验。

状态：.cursor/goal-abl-dynamic.json
用法：
  python scripts/eddy_ablation_dynamic.py init
  python scripts/eddy_ablation_dynamic.py status
  python scripts/eddy_ablation_dynamic.py next          # 打印下一 profile 或 DONE
  python scripts/eddy_ablation_dynamic.py record --profile 4_bgr_zeta
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.config import resolve_path

STATE_PATH = REPO / ".cursor" / "goal-abl-dynamic.json"
REPORT_PATH = REPO / "submission" / "tables" / "eddy_ablation_dynamic_plan.md"

SINGLES = ("4_bgr_zeta", "4_bgr_ow", "5_bgr_grad")
SINGLE_LABEL = {
    "4_bgr_zeta": "zeta",
    "4_bgr_ow": "ow",
    "5_bgr_grad": "grad",
}
PAIRS = (
    ("5_no_grad", ("zeta", "ow")),
    ("6_no_ow", ("zeta", "grad")),
    ("6_no_zeta", ("ow", "grad")),
)

DELTA_HELPFUL = 0.005
DELTA_HARMFUL = -0.02


def _iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_map50(metrics_path: Path, split: str) -> float | None:
    p = metrics_path if split == "val" else metrics_path.parent / metrics_path.name.replace("_val", f"_{split}")
    if split == "val":
        p = metrics_path
    else:
        p = metrics_path.parent / f"metrics_summary_{split}.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    v = (data.get("metrics") or {}).get("mask_map50")
    return float(v) if v is not None else None


def _load_metrics(out_rel: str) -> dict[str, float | None]:
    base = resolve_path(out_rel)
    return {
        "val_map50": _read_map50(base / "metrics_summary_val.json", "val"),
        "test_map50": _read_map50(base / "metrics_summary_test.json", "test"),
    }


def _classify_delta(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    if delta >= DELTA_HELPFUL:
        return "helpful"
    if delta <= DELTA_HARMFUL:
        return "harmful"
    return "neutral"


def _save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _iso()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(state)


def _write_report(state: dict) -> None:
    b = state.get("baseline") or {}
    lines = [
        "# 3ch 增通道消融 — 动态规划（自动生成）",
        "",
        f"更新：{state.get('updated_at', '')}",
        "",
        f"**基线 3ch**：val={b.get('val_map50')} test={b.get('test_map50')}",
        "",
        "## 已完成",
        "",
        "| profile | val | test | Δval | 判定 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in sorted((state.get("completed") or {}).items()):
        lines.append(
            f"| {name} | {row.get('val_map50')} | {row.get('test_map50')} | "
            f"{row.get('delta_val')} | {row.get('verdict')} |"
        )
    skipped = state.get("skipped") or {}
    if skipped:
        lines.extend(["", "## 已跳过", ""])
        for name, why in sorted(skipped.items()):
            lines.append(f"- **{name}**：{why}")
    q = state.get("queue") or []
    lines.extend(["", "## 待跑队列", ""])
    if q:
        for i, p in enumerate(q, 1):
            lines.append(f"{i}. `{p}`")
    else:
        lines.append("（空 — 规划结束或等待 record 后重算）")
    nxt = state.get("next")
    if nxt:
        lines.extend(["", f"**下一实验**：`{nxt}`", f"阶段：{state.get('phase')}", ""])
    reason = state.get("last_plan_reason")
    if reason:
        lines.extend(["", f"**规划说明**：{reason}", ""])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_init() -> int:
    m3 = _load_metrics("outputs/eddy_cloud_fair")
    m7 = _load_metrics("outputs/eddy_enh7_cloud_fair")
    if m3["val_map50"] is None:
        print("ERROR: 缺少 outputs/eddy_cloud_fair/metrics_summary_val.json，请先跑 3ch 基线 eval")
        return 1

    completed: dict = {
        "3ch_baseline": {
            **m3,
            "delta_val": 0.0,
            "verdict": "baseline",
            "source": "outputs/eddy_cloud_fair",
        },
    }
    skipped: dict = {}
    if m7["val_map50"] is not None:
        dv7 = m7["val_map50"] - m3["val_map50"]
        completed["7ch_full_prior"] = {
            **m7,
            "delta_val": round(dv7, 6),
            "verdict": _classify_delta(dv7),
            "source": "outputs/eddy_enh7_cloud_fair",
            "note": "已有公平 7ch 权重，不重复训练除非动态规划要求",
        }

    state = {
        "version": 1,
        "method": "3ch_incremental_dynamic",
        "baseline": {
            "profile": "3ch",
            "val_map50": m3["val_map50"],
            "test_map50": m3["test_map50"],
        },
        "phase": "singles",
        "queue": list(SINGLES),
        "completed": completed,
        "skipped": skipped,
        "thresholds": {
            "helpful_delta_val": DELTA_HELPFUL,
            "harmful_delta_val": DELTA_HARMFUL,
        },
        "next": SINGLES[0],
        "last_plan_reason": "初始化：先跑三路单通道叠加（ζ / OW / Grad）",
        "created_at": _iso(),
    }
    _save(state)
    print(f"init -> next={state['next']}")
    print(f"wrote {STATE_PATH.relative_to(REPO)}")
    print(f"wrote {REPORT_PATH.relative_to(REPO)}")
    return 0


def _baseline_val(state: dict) -> float:
    return float(state["baseline"]["val_map50"])


def _record_profile(state: dict, profile: str) -> None:
    out_rel = f"outputs/eddy_ablation/{profile}"
    m = _load_metrics(out_rel)
    if m["val_map50"] is None:
        raise SystemExit(f"ERROR: 无指标 {out_rel}/metrics_summary_val.json，请先 train+eval")

    bv = _baseline_val(state)
    dv = m["val_map50"] - bv
    state.setdefault("completed", {})[profile] = {
        **m,
        "delta_val": round(dv, 6),
        "verdict": _classify_delta(dv),
        "source": out_rel,
    }
    q = list(state.get("queue") or [])
    if profile in q:
        q.remove(profile)
    state["queue"] = q


def _single_verdicts(state: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for prof, label in SINGLE_LABEL.items():
        row = (state.get("completed") or {}).get(prof)
        if row:
            out[label] = row.get("verdict") or _classify_delta(row.get("delta_val"))
    return out


def _plan_pairs(state: dict) -> tuple[list[str], str]:
    sv = _single_verdicts(state)
    if len(sv) < 3:
        missing = [SINGLE_LABEL[p] for p in SINGLES if p not in (state.get("completed") or {})]
        return list(state.get("queue") or []), f"单路未齐，缺: {missing}"

    queue: list[str] = []
    skipped = dict(state.get("skipped") or {})
    reasons: list[str] = []

    for pair_prof, (a, b) in PAIRS:
        va, vb = sv.get(a, "unknown"), sv.get(b, "unknown")
        if va == "harmful" and vb == "harmful":
            skipped[pair_prof] = f"{a}、{b} 单路均有害，跳过"
            continue
        if va == "harmful" and vb != "helpful":
            skipped[pair_prof] = f"{a} 有害且 {b} 未达 helpful，跳过"
            continue
        if vb == "harmful" and va != "helpful":
            skipped[pair_prof] = f"{b} 有害且 {a} 未达 helpful，跳过"
            continue
        queue.append(pair_prof)
        reasons.append(f"保留 {pair_prof}（{a}={va}, {b}={vb}）")

    state["skipped"] = skipped
    state["phase"] = "pairs"

    best_dv = max(
        (r.get("delta_val") or -1.0 for r in (state.get("completed") or {}).values() if isinstance(r.get("delta_val"), (int, float))),
        default=-1.0,
    )
    if best_dv >= -0.01 and "7ch_full_prior" not in (state.get("completed") or {}):
        reasons.append("组合阶段：已有 7ch 先验结果，不重训 7ch")
    elif best_dv < DELTA_HARMFUL and not queue:
        reasons.append("所有单路显著劣于基线，建议结束增通道实验")
        state["phase"] = "done"

    reason = "; ".join(reasons) if reasons else "双路组合队列为空"
    return queue, reason


def _replan(state: dict) -> None:
    phase = state.get("phase", "singles")
    q = list(state.get("queue") or [])

    if phase == "singles":
        remaining = [p for p in SINGLES if p not in (state.get("completed") or {})]
        if remaining:
            state["queue"] = remaining
            state["next"] = remaining[0]
            state["last_plan_reason"] = f"继续单路：{remaining[0]}"
            return
        state["phase"] = "pairs"
        q, reason = _plan_pairs(state)
        state["queue"] = q
        state["next"] = q[0] if q else None
        state["last_plan_reason"] = reason
        if not q:
            state["phase"] = "done"
        return

    if phase == "pairs":
        remaining = [p for p in (state.get("queue") or []) if p not in (state.get("completed") or {})]
        if remaining:
            state["queue"] = remaining
            state["next"] = remaining[0]
            state["last_plan_reason"] = f"继续双路：{remaining[0]}"
            return
        state["phase"] = "done"
        state["next"] = None
        state["queue"] = []
        state["last_plan_reason"] = "双路队列已跑完；查看表 eddy_ablation_dynamic_plan.md 与 matrix"
        return

    state["next"] = None
    state["queue"] = []


def cmd_record(profile: str) -> int:
    if not STATE_PATH.is_file():
        print("ERROR: 先运行 init")
        return 1
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    _record_profile(state, profile)
    _replan(state)
    _save(state)
    nxt = state.get("next")
    print(f"record {profile} -> next={nxt or 'DONE'}")
    print(state.get("last_plan_reason", ""))
    return 0


def cmd_next() -> int:
    if not STATE_PATH.is_file():
        cmd_init()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    nxt = state.get("next")
    if not nxt:
        print("DONE")
        return 0
    print(nxt)
    return 0


def cmd_status() -> int:
    if not STATE_PATH.is_file():
        print("未初始化；运行: python scripts/eddy_ablation_dynamic.py init")
        return 1
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    print(json.dumps(
        {
            "phase": state.get("phase"),
            "next": state.get("next"),
            "queue": state.get("queue"),
            "last_plan_reason": state.get("last_plan_reason"),
            "baseline": state.get("baseline"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("next")
    rec = sub.add_parser("record")
    rec.add_argument("--profile", required=True)
    args = ap.parse_args()

    if args.cmd == "init":
        return cmd_init()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "next":
        return cmd_next()
    if args.cmd == "record":
        return cmd_record(args.profile)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
