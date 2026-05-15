#!/usr/bin/env python3
"""Generate validation JSONL disjoint from anomaly_llm_sft_seed_v1_fixed.jsonl (same ChatML schema)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _train_candidate_event_ids(train_path: Path) -> set[str]:
    out: set[str] = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        for m in obj.get("messages", []):
            if m.get("role") != "user":
                continue
            payload = json.loads(m["content"])
            for c in payload.get("typhoon_link", {}).get("candidates") or []:
                eid = c.get("event_id")
                if eid:
                    out.add(str(eid))
            break
    return out


def main() -> None:
    events_path = REPO_ROOT / "data/processed/anomaly/typhoon_kb/events.json"
    train_path = REPO_ROOT / "submission/datasets/anomaly_llm_sft_seed_v1_fixed.jsonl"
    out_path = REPO_ROOT / "submission/datasets/anomaly_llm_sft_val_seed_v1.jsonl"
    out_min = REPO_ROOT / "submission/datasets/anomaly_llm_sft_val_seed_v1_chatml_min.jsonl"

    events = json.loads(events_path.read_text(encoding="utf-8"))
    train_eids = _train_candidate_event_ids(train_path)

    # Index ranges disjoint from train slices [0:25:2], [200:225:2], [500:525:2]
    picked: list[dict] = []
    for start, stop in ((100, 125), (300, 325), (600, 625)):
        picked.extend(events[i] for i in range(start, stop, 2))
    picked = picked[:10]

    system = (
        "你是海洋风浪监测报告助手。必须基于输入事实，不得编造台风事件。"
        "输出严格JSON，包含summary_anomaly、impact、historical_analogy、actions。"
    )

    rows: list[dict] = []
    for i, e in enumerate(picked, start=1):
        st = str(e.get("start_time", ""))
        et = str(e.get("end_time", ""))
        lon_min = float(e.get("lon_min", 110.0))
        lon_max = float(e.get("lon_max", 130.0))
        lat_min = float(e.get("lat_min", 10.0))
        lat_max = float(e.get("lat_max", 30.0))
        level = "high" if i % 3 == 0 else ("medium" if i % 3 == 1 else "low")

        anomaly: dict = {
            "anomaly_level": level,
            "anomaly_index": round(0.3 + 0.03 * i, 3),
            "wind_z": round(0.85 + 0.07 * (i % 8), 3),
            "wave_z": round(0.75 + 0.09 * (i % 9), 3),
            "threshold_rule": "3sigma",
        }
        if i in (2, 9):
            anomaly["assessment_note"] = "DTW不可用，已进入无曲线降级模式"
            anomaly["wave_z"] = None

        candidates: list[dict] = []
        if i % 4 != 0:
            candidates = [
                {
                    "event_id": e.get("event_id"),
                    "name": e.get("name", ""),
                    "start_time": st,
                    "end_time": et,
                    "score": round(40.0 + 0.65 * i, 3),
                    "dtw_distance": round(1.15 + 0.035 * i, 3),
                }
            ]

        inp = {
            "anomaly_result": anomaly,
            "typhoon_link": {
                "query": {
                    "start_time": st,
                    "end_time": et,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                },
                "candidates": candidates,
            },
        }

        if candidates:
            eid = candidates[0]["event_id"]
            out_obj = {
                "summary_anomaly": (
                    "存在异常信号，建议提高监测等级。"
                    if level != "low"
                    else "当前信号偏弱，建议常规跟踪。"
                ),
                "impact": "可能对近海作业与航线稳定性造成扰动，需滚动复核。",
                "historical_analogy": (
                    f"与知识库事件 {eid} 在时空窗口上存在一定可比性，但需结合实测进一步确认。"
                ),
                "actions": [
                    "未来6小时持续监测风速与波高",
                    "结合业务阈值进行人工复核",
                    "若异常持续抬升则升级响应",
                ],
            }
        else:
            out_obj = {
                "summary_anomaly": (
                    "当前窗口检测到异常迹象，但知识库未命中可比台风。"
                    if level != "low"
                    else "当前窗口总体平稳，未见明确台风关联。"
                ),
                "impact": "无历史台风命中不代表无风险，应结合后续观测判断演化趋势。",
                "historical_analogy": (
                    "在当前检索范围内未发现足够重叠的历史台风样本，可作为非台风主导或证据不足情形处理。"
                ),
                "actions": [
                    "未来6小时持续监测风速与波高",
                    "结合业务阈值进行人工复核",
                    "若异常持续抬升则升级响应",
                ],
            }

        if anomaly.get("assessment_note"):
            out_obj["impact"] += "（请注意：观测曲线部分缺失，类比与建议需偏重监测与复核。）"

        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(inp, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(out_obj, ensure_ascii=False)},
                ]
            }
        )

    val_eids: set[str] = set()
    for r in rows:
        for m in r["messages"]:
            if m.get("role") != "user":
                continue
            payload = json.loads(m["content"])
            for c in payload.get("typhoon_link", {}).get("candidates") or []:
                eid = c.get("event_id")
                if eid:
                    val_eids.add(str(eid))
            break
    overlap = train_eids & val_eids
    if overlap:
        raise SystemExit(f"train/val candidate event_id overlap: {overlap}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with out_min.open("w", encoding="utf-8", newline="\n") as fo:
        for r in rows:
            msgs = r["messages"]
            user = next(x for x in msgs if x["role"] == "user")
            assistant = next(x for x in msgs if x["role"] == "assistant")
            mini = {
                "messages": [
                    {"role": "user", "content": user["content"]},
                    {"role": "assistant", "content": assistant["content"]},
                ]
            }
            fo.write(json.dumps(mini, ensure_ascii=False) + "\n")

    n_with_cand = sum(
        1
        for r in rows
        if json.loads(next(m["content"] for m in r["messages"] if m["role"] == "user"))[
            "typhoon_link"
        ]["candidates"]
    )
    print(f"wrote {out_path}")
    print(f"wrote {out_min}")
    print(f"samples={len(rows)} with_candidates={n_with_cand} no_candidates={len(rows) - n_with_cand}")
    assert len(rows) == 10


if __name__ == "__main__":
    main()
