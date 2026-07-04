#!/usr/bin/env python3
"""V6 生命周期统计：OW(P24) 伪标签相邻日跟踪，只统计不筛标。"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eddy_ablation_common import (  # noqa: E402
    Instance,
    instance_area_px,
    load_samples,
    mask_iou,
    read_label,
    resolve_path,
)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def _match_tracks(
    prev: list[Instance],
    curr: list[Instance],
    *,
    iou_thr: float = 0.3,
) -> list[tuple[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(prev):
        for ci, c in enumerate(curr):
            if p.cls != c.cls:
                continue
            iou = mask_iou(p.mask, c.mask)
            if iou >= iou_thr:
                pairs.append((iou, pi, ci))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_c: set[int] = set()
    out: list[tuple[int, int]] = []
    for _iou, pi, ci in pairs:
        if pi in used_p or ci in used_c:
            continue
        used_p.add(pi)
        used_c.add(ci)
        out.append((pi, ci))
    return out


def _lifetime_bucket(days: int) -> str:
    if days <= 1:
        return "1"
    if days == 2:
        return "2"
    if days <= 5:
        return "3~5"
    return ">5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="AutoDL/dataset/eddy_v6_leakage")
    ap.add_argument("--nc-root", default="服创数据集/中尺度涡识别")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-md", default="submission/tables/eddy_v6_lifetime_val.md")
    args = ap.parse_args()

    root = resolve_path(args.dataset_root)
    samples = load_samples(root, None, resolve_path(args.nc_root), [args.split])
    by_nc: dict[str, list] = defaultdict(list)
    for s in samples:
        by_nc[s.nc_stem].append(s)
    for stem in by_nc:
        by_nc[stem].sort(key=lambda x: x.time_idx)

    lifetimes: list[int] = []
    for _stem, seq in by_nc.items():
        track_len: dict[int, int] = {}
        track_cls: dict[int, int] = {}
        next_tid = 0
        prev_map: dict[int, int] = {}  # inst_idx -> track_id
        for si, sample in enumerate(seq):
            label_path = root / "labels" / args.split / f"{sample.png.stem}.txt"
            h, w = sample.rgb.shape[:2]
            curr = read_label(label_path, h, w)
            if si == 0:
                for ci, inst in enumerate(curr):
                    prev_map[ci] = next_tid
                    track_len[next_tid] = 1
                    track_cls[next_tid] = inst.cls
                    next_tid += 1
                continue
            prev = read_label(
                root / "labels" / args.split / f"{seq[si - 1].png.stem}.txt", h, w
            )
            matches = _match_tracks(prev, curr)
            matched_c: set[int] = set()
            new_prev: dict[int, int] = {}
            for pi, ci in matches:
                tid = prev_map.get(pi)
                if tid is None:
                    continue
                track_len[tid] = track_len.get(tid, 0) + 1
                new_prev[ci] = tid
                matched_c.add(ci)
            for ci, inst in enumerate(curr):
                if ci in matched_c:
                    continue
                new_prev[ci] = next_tid
                track_len[next_tid] = 1
                track_cls[next_tid] = inst.cls
                next_tid += 1
            prev_map = new_prev
        lifetimes.extend(track_len.values())

    if not lifetimes:
        raise SystemExit("无轨迹统计")

    buckets = {_lifetime_bucket(d): 0 for d in [1, 2, 3, 4, 6, 10]}
    for d in lifetimes:
        buckets[_lifetime_bucket(d)] = buckets.get(_lifetime_bucket(d), 0) + 1
    short_frac = sum(1 for d in lifetimes if d == 1) / len(lifetimes)

    lines = [
        "# V6 OW(P24) 伪标签生命周期（val）",
        "",
        f"- dataset: `{root}`",
        f"- split: {args.split}",
        f"- tracks: {len(lifetimes)}",
        f"- mean lifetime (days): {np.mean(lifetimes):.2f}",
        f"- Lifetime=1 占比: {short_frac:.1%}",
        "",
        "| bucket | count |",
        "| --- | ---: |",
    ]
    for k in ("1", "2", "3~5", ">5"):
        lines.append(f"| {k} | {buckets.get(k, 0)} |")
    lines.append("")
    lines.append("说明：仅统计，不参与标签筛选。")

    out = resolve_path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
