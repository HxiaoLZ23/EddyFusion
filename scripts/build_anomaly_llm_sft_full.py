#!/usr/bin/env python3
"""
构建风浪异常 LLM（百炼）SFT 正式训练集 / 验证集 ChatML(JSONL)。
与 docs/开发规划/风浪异常模块_LLM演示报告优化_百炼调优规划.md §9 对齐：烟测复盘后的 system 约束 + 分层场景。

产出（默认路径，第二轮）：
  submission/datasets/anomaly_llm_sft_train_chatml_r2.jsonl   (~300条)
  submission/datasets/anomaly_llm_sft_val_chatml_r2.jsonl     (~50条)

用法：
  python scripts/build_anomaly_llm_sft_full.py

依赖：data/processed/anomaly/typhoon_kb/events.json（先跑 scripts/run_typhoon_kb.ps1）

第二轮（默认）：输出 `*_chatml_r2.jsonl`，收紧无依据概率、degrade+候选一致、完整 JSON。
第一轮文件仍保留为 `anomaly_llm_sft_*_chatml.jsonl`（可用 --out-train 覆盖写回）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.llm_sft_report_prompt import SYSTEM_CHATML_ANOMALY_REPORT as SYSTEM_CHATML

TRAIN_LINES = 300
VAL_LINES = 50


def _span_query(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_time": str(e.get("start_time", "")),
        "end_time": str(e.get("end_time", "")),
        "lon_min": float(e.get("lon_min", 0.0)),
        "lon_max": float(e.get("lon_max", 0.0)),
        "lat_min": float(e.get("lat_min", 0.0)),
        "lat_max": float(e.get("lat_max", 0.0)),
    }


def _bucket_for_line(line_idx: int) -> str:
    """约 40% no / 38% ty / 12% degrade / 10% edge（每 50 条线为一周期）."""
    k = line_idx % 50
    if k < 20:
        return "no_typhoon"
    if k < 39:
        return "with_typhoon"
    if k < 46:
        return "degrade"
    return "edge"


def _synthetic_residuals(seed: int) -> tuple[dict[str, Any], str, str]:
    """返回 (anomaly_result, anomaly_level, 仅用于教官回复拼接的 reconcile 句，不写进用户 JSON)."""
    rnd = random.Random(seed)
    level = rnd.choice(("low", "medium", "high"))
    anomaly_index = round(0.22 + rnd.random() * 0.62, 3)
    wind_z = round(0.4 + rnd.random() * 3.8, 3)
    wave_z = round(0.35 + rnd.random() * 3.5, 3)
    wind_residual = round(0.2 + rnd.random() * 2.2, 3)
    wave_residual = round(0.05 + rnd.random() * 1.2, 3)
    anomaly: dict[str, Any] = {
        "anomaly_level": level,
        "anomaly_index": anomaly_index,
        "wind_residual": wind_residual,
        "wind_z": wind_z,
        "wave_residual": wave_residual,
        "wave_z": wave_z,
        "threshold_rule": "3sigma",
    }

    reconcile = ""
    if level == "high" and max(wind_z, wave_z) < 2.0:
        reconcile = "综合上看，分项 z_score 与「高等级」语感可能不完全对齐，请以 anomaly_index 与阈值规则的联动解释为准。"
        anomaly["threshold_rule"] = "3sigma+index"
    if level == "low" and max(wind_z, wave_z) > 3.0:
        reconcile = "分项 z_score 存在抬升但整体等级仍为偏低判定，请以 anomaly_level / anomaly_index 为主文案依据。"

    return anomaly, level, reconcile


def _assistant_typhoon(
    bucket: str,
    e: dict[str, Any],
    anomaly: dict[str, Any],
    *,
    reconcile_hint: str = "",
    top_cand: dict[str, Any] | None = None,
) -> dict[str, Any]:
    st = _span_query(e)
    eid = e.get("event_id", "")
    name = str(e.get("name", "") or "").strip()

    level = anomaly.get("anomaly_level", "medium")
    idx_s = anomaly.get("anomaly_index", 0)

    impacts = []
    impacts.append(f"在时间窗「{st['start_time']}」至「{st['end_time']}」、海区 lon[{st['lon_min']:.1f},{st['lon_max']:.1f}] "
                   f"lat[{st['lat_min']:.1f},{st['lat_max']:.1f}] 内给出异常研判。")
    if bucket == "degrade" and top_cand:
        eid_c = top_cand.get("event_id", "")
        nm_c = str(top_cand.get("name", "") or "").strip()
        sc = top_cand.get("score", "")
        impacts.insert(
            0,
            f"检索已返回弱候选（event_id={eid_c}"
            + (f"，名称片段「{nm_c[:24]}」" if nm_c else "")
            + f"，score={sc}）；不得以「知识库未收录致因」否定候选存在。"
            "波高侧降级时，风浪协同证据链不完整，影响判断应标为「中高不确定」。",
        )
    impacts.append(
        "影响判断带不确定性：请结合实况观测与业务应急预案滚动更新；本节为辅助说明。"
    )

    if bucket == "no_typhoon":
        sum_a = (
            f"综合判定等级为「{level}」（anomaly_index≈{idx_s:.2f}），"
            f"风速残差偏高（z-score 相对较高），波高也出现相对抬升。"
        )
        if level == "low":
            sum_a = (
                f"综合判定偏向「偏低关注」（anomaly_index≈{idx_s:.2f}），"
                f"分项残差存在一定波动但整体未达高等级。"
            )
        if reconcile_hint:
            sum_a += " " + reconcile_hint

        hist = (
            "在当前给定时间窗与海区范围的台风查询检索结果中，未返回可对齐的历史台风个例。"
            "这仅表示在本次索引与阈值下未发现足够重叠的类比对象，并不等于海区一定无风浪风险，"
            "亦不排除温带系统、局地强对流或其它致因。"
        )

    elif bucket == "with_typhoon":
        sum_a = (
            f"综合判定等级「{level}」（anomaly_index≈{idx_s:.2f}），并与候选台风事件形成潜在关联。"
        )
        if reconcile_hint:
            sum_a += " " + reconcile_hint
        if name:
            sum_a += f" 最接近的检索候选为 {name}（{eid}）。"
        else:
            sum_a += f" 最接近的检索候选编号为 {eid}。"
        hist = (
            f"类比仅基于检索返回的候选 {eid}："
            f"score 为检索计分项，time_overlap_hours 与 bbox_overlap_ratio 反映时空重叠程度，"
            f"dtw_distance 为序列相似度辅助项（若为空则未参与或未计算）。"
            "以上字段均以用户 JSON 为准，不据此推断历史统计概率或外域天气规律。"
        )

    elif bucket == "degrade":
        anomaly["wave_z"] = None
        anomaly["assessment_note"] = (
            "波高观测曲线缺失或未通过质量检查，风浪联动进入无曲线降级；"
            "风速序列仍可用于粗略对照，但整体置信度下降。"
        )
        note = str(anomaly.get("assessment_note", ""))
        sum_a = (
            f"当前窗口综合判定为「{level}」（anomaly_index≈{idx_s:.2f}），风速分项信号相对突出。"
            f"数据局限：{note}"
        )
        if reconcile_hint:
            sum_a += " " + reconcile_hint
        sc_txt = ""
        if top_cand:
            sc_txt = f"弱候选检索分 score={top_cand.get('score')}。"
        hist = (
            f"检索返回弱候选 {eid}（{name or '未命名'}）。{sc_txt}"
            "因波高侧观测链路降级，不宜将候选强度与当前波高异常做强绑定；"
            "类比仅限候选时间与空间字段与当前 query 的可比性，不做致因概率推断。"
        )

    else:
        eid_safe = str(eid) if eid is not None else "unknown"
        anomaly["wind_z"] = round(3.2 + random.Random(abs(hash(eid_safe)) % (2**32)).random() * 0.9, 3)
        anomaly["anomaly_level"] = "medium"
        level = "medium"
        sum_a = (
            "监测显示异常特征处于「中等」等级，分项风速侧的相对偏离较明显；"
            "综合指数与分项之间可能存在体感差异，需以规则侧的 anomaly_index 与等级标签为准进行二次理解。"
        )
        if reconcile_hint:
            sum_a += " " + reconcile_hint
        hist = (
            "本条为边角合成场景：请以输入中的 anomaly_level 与 anomaly_index 为主语义，不把单条 z_score 当作唯一裁决。"
        )

    acts = [
        "未来 6～12 小时加密查看风速与波高序列与外场简报",
        "按业务阈值启动人工复核与留痕归档",
        "若等级抬升或与预案触发条件对齐，升级到值班/会商流程",
    ]
    if bucket == "no_typhoon":
        acts.append("扩大检索或使用相邻时间窗做二次联动，核对是否为检索边界导致空结果")
    if bucket == "degrade":
        acts.append("优先补全或复核波高观测曲线，再决定是否收紧风浪联合阈值")

    impact = "".join(s for s in impacts if s).strip()

    return {
        "summary_anomaly": sum_a.strip(),
        "impact": impact,
        "historical_analogy": hist.strip(),
        "actions": acts,
    }


def _samples_for_chunk(
    events: list[dict[str, Any]],
    indices: list[int],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, ei in enumerate(indices):
        # 打散事件用于 with_typhoon 时减少相邻重复形态
        e = events[ei]
        bucket = _bucket_for_line(line_no)
        anomaly, _level_syn, reconcile = _synthetic_residuals(rng.randint(1, 10**9))

        cand: list[dict[str, Any]] = []
        if bucket in ("with_typhoon", "degrade", "edge"):
            cand = [
                {
                    "event_id": e.get("event_id"),
                    "name": e.get("name", ""),
                    "start_time": e.get("start_time"),
                    "end_time": e.get("end_time"),
                    "intensity_level": e.get("intensity_level"),
                    "peak_wind_kt": e.get("peak_wind_kt"),
                    "score": round(20.0 + (line_no % 17) * 3.15, 3),
                    "bbox_overlap_ratio": round(0.12 + rng.random() * 0.5, 4),
                    "time_overlap_hours": round(8.0 + rng.random() * 40.0, 3),
                    "dtw_distance": round(1.0 + rng.random() * 0.35, 3),
                }
            ]
        ty_cand: list = [] if bucket == "no_typhoon" else cand
        top = ty_cand[0] if ty_cand else None

        out_obj = _assistant_typhoon(bucket, e, anomaly, reconcile_hint=reconcile, top_cand=top)
        payload: dict[str, Any] = {
            "anomaly_result": anomaly,
            "typhoon_link": {"query": _span_query(e), "candidates": ty_cand},
        }

        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_CHATML},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(out_obj, ensure_ascii=False)},
            ]
        }
        rows.append(row)
    return rows


def _build_index_lists(n_events: int, train_lines: int, val_lines: int, seed: int) -> tuple[list[int], list[int]]:
    need = train_lines + val_lines
    if n_events < need:
        raise SystemExit(f"events.json 条数过少: {n_events} < {need}")
    rnd = random.Random(seed)
    pool = rnd.sample(range(n_events), need)
    return pool[val_lines:], pool[:val_lines]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--train-lines", type=int, default=TRAIN_LINES)
    parser.add_argument("--val-lines", type=int, default=VAL_LINES)
    parser.add_argument(
        "--out-train",
        type=str,
        default="submission/datasets/anomaly_llm_sft_train_chatml_r2.jsonl",
    )
    parser.add_argument(
        "--out-val",
        type=str,
        default="submission/datasets/anomaly_llm_sft_val_chatml_r2.jsonl",
    )
    parser.add_argument(
        "--out-train-min",
        type=str,
        default="submission/datasets/anomaly_llm_sft_train_chatml_min_r2.jsonl",
    )
    parser.add_argument(
        "--out-val-min",
        type=str,
        default="submission/datasets/anomaly_llm_sft_val_chatml_min_r2.jsonl",
    )
    args = parser.parse_args()

    events_path = REPO_ROOT / "data/processed/anomaly/typhoon_kb/events.json"
    if not events_path.is_file():
        print(f"缺失 {events_path}，请先构建台风查询索引。", file=sys.stderr)
        raise SystemExit(1)

    events = json.loads(events_path.read_text(encoding="utf-8"))
    n_ev = len(events)
    rnd = random.Random(args.seed)

    train_ix, val_ix = _build_index_lists(n_ev, args.train_lines, args.val_lines, args.seed)

    train_rows = _samples_for_chunk(events, train_ix, rng=rnd)
    val_rows = _samples_for_chunk(events, val_ix, rng=random.Random(args.seed + 1))

    out_train = REPO_ROOT / args.out_train
    out_val = REPO_ROOT / args.out_val
    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_val, val_rows)

    def to_min(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out_min: list[dict[str, Any]] = []
        for r in rows:
            msgs = r["messages"]
            u = next(m for m in msgs if m["role"] == "user")
            a = next(m for m in msgs if m["role"] == "assistant")
            out_min.append({"messages": [{"role": "user", "content": u["content"]}, {"role": "assistant", "content": a["content"]}]})
        return out_min

    _write_jsonl(REPO_ROOT / args.out_train_min, to_min(train_rows))
    _write_jsonl(REPO_ROOT / args.out_val_min, to_min(val_rows))

    def count_buckets(rows: list[dict[str, Any]]) -> dict[str, int]:
        ctr: dict[str, int] = {}
        for i in range(len(rows)):
            b = _bucket_for_line(i)
            ctr[b] = ctr.get(b, 0) + 1
        return ctr

    print(f"events_total={n_ev}")
    print(f"wrote {out_train} n={len(train_rows)} buckets={count_buckets(train_rows)}")
    print(f"wrote {out_val} n={len(val_rows)} buckets={count_buckets(val_rows)}")
    print(f"wrote {REPO_ROOT / args.out_train_min}")
    print(f"wrote {REPO_ROOT / args.out_val_min}")


if __name__ == "__main__":
    main()
