#!/usr/bin/env python3
"""验收 V6 Phase B0 各 dataset 的 train/val stem 完全一致，并写 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.config import resolve_path


def _stems(root: Path, split: str) -> frozenset[str]:
    img = root / "images" / split
    if not img.is_dir():
        return frozenset()
    return frozenset(p.stem for p in img.glob("*.png"))


def _labels_hash(root: Path, split: str, stems: frozenset[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(stems):
        p = root / "labels" / split / f"{s}.txt"
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="逗号分隔 dataset 根")
    ap.add_argument("--manifest", default="submission/tables/eddy_v6_b0_stem_manifest.json")
    ap.add_argument("--expect-train", type=int, default=355)
    ap.add_argument("--expect-val", type=int, default=356)
    args = ap.parse_args()

    roots = [resolve_path(x.strip()) for x in args.datasets.split(",") if x.strip()]
    if len(roots) < 2:
        raise SystemExit("至少 2 个 dataset")

    manifest: dict = {"datasets": [str(r) for r in roots], "splits": {}}
    ok = True
    for split in ("train", "val"):
        sets = [_stems(r, split) for r in roots]
        ref = sets[0]
        for i, s in enumerate(sets[1:], start=1):
            if s != ref:
                ok = False
                only0 = ref - s
                only1 = s - ref
                print(f"ERROR {split}: {roots[0].name} vs {roots[i].name} stem 不一致")
                print(f"  only in first: {len(only0)}  only in other: {len(only1)}")
        exp = args.expect_train if split == "train" else args.expect_val
        n = len(ref)
        if n != exp:
            ok = False
            print(f"ERROR {split}: n={n} (expect {exp})")
        manifest["splits"][split] = {
            "n_stems": n,
            "expect": exp,
            "labels_hash": _labels_hash(roots[0], split, ref),
            "stems_sample": sorted(ref)[:3],
        }

    out = resolve_path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest["train_stems"] = sorted(_stems(roots[0], "train"))
    manifest["val_stems"] = sorted(_stems(roots[0], "val"))
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if not ok:
        raise SystemExit(1)
    print(f"OK train={len(_stems(roots[0], 'train'))} val={len(_stems(roots[0], 'val'))}")


if __name__ == "__main__":
    main()
