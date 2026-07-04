#!/usr/bin/env python3
"""在已有 3ch ``data/processed/eddy`` 划分上复制 PNG/标签并写入 7ch ``.npy``（同帧对比）。"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> None:
    from src.eddy.stacked_physics import build_physics_stacked_hw7
    from src.preprocess.eddy_physics import okubo_weiss_and_vorticity
    from src.utils.config import project_root, resolve_path
    from src.utils.xarray_nc_open import open_xr_dataset_compat

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/eddy")
    ap.add_argument("--dst", default="data/processed/eddy_enh7")
    ap.add_argument(
        "--yaml-path",
        default=None,
        help="写入 dataset.yaml 的 path 字段（相对仓库根）；默认与 --dst 一致",
    )
    ap.add_argument(
        "--nc-root",
        default="服创数据集/中尺度涡识别",
        help="按 png stem 匹配同名 nc",
    )
    args = ap.parse_args()

    src = resolve_path(args.src)
    dst = resolve_path(args.dst)
    nc_root = resolve_path(args.nc_root)

    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}")

    for sp in ("train", "val", "test"):
        for sub in ("images", "labels"):
            s = src / sub / sp
            d = dst / sub / sp
            if s.is_dir():
                d.mkdir(parents=True, exist_ok=True)
                for f in s.iterdir():
                    if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".txt"):
                        tgt = d / f.name
                        if not tgt.exists():
                            shutil.copy2(f, tgt)

    rel = args.yaml_path
    if not rel:
        try:
            rel = dst.resolve().relative_to(project_root().resolve()).as_posix()
        except ValueError:
            rel = dst.as_posix()
    ds_yaml = (
        f"path: {rel}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "channels: 7\n"
        "names:\n"
        "  0: eddy_cyclonic\n"
        "  1: eddy_anticyclonic\n"
    )
    (dst / "dataset.yaml").write_text(ds_yaml, encoding="utf-8")

    nc_cache: dict[str, tuple] = {}
    n_ok = n_skip = 0

    def _pick_da(ds, names):
        lower = {str(k).lower(): k for k in ds.data_vars}
        for n in names:
            if n.lower() in lower:
                return ds[lower[n.lower()]]
        raise KeyError(names)

    for sp in ("train", "val", "test"):
        img_dir = dst / "images" / sp
        for png in sorted(img_dir.glob("*.png")):
            m = re.match(r"^(.+)_t(\d+)$", png.stem)
            if not m:
                n_skip += 1
                continue
            nc_stem, t_idx = m.group(1), int(m.group(2))
            nc_path = nc_root / f"{nc_stem}.nc"
            if not nc_path.is_file():
                n_skip += 1
                continue
            npy_p = png.with_suffix(".npy")
            if npy_p.is_file():
                n_ok += 1
                continue
            if nc_stem not in nc_cache:
                ds, tmp = open_xr_dataset_compat(nc_path)
                nc_cache[nc_stem] = (ds, tmp, nc_path)
            else:
                ds, tmp, nc_path = nc_cache[nc_stem]
            try:
                adt = _pick_da(ds, ("adt", "ADT"))
                ug = _pick_da(ds, ("ugos", "UGOS"))
                vg = _pick_da(ds, ("vgos", "VGOS"))
                _sp = {"latitude", "longitude", "lat", "lon"}
                tdim = [d for d in adt.dims if d not in _sp][0]
                a = np.asarray(adt.isel({tdim: t_idx}).values, dtype=np.float64)
                u = np.asarray(ug.isel({tdim: t_idx}).values, dtype=np.float64)
                v = np.asarray(vg.isel({tdim: t_idx}).values, dtype=np.float64)
                lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
                lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
                zeta, ow = okubo_weiss_and_vorticity(u, v, lat, lon)
                stack = build_physics_stacked_hw7(a, u, v, zeta, ow)
                np.save(str(npy_p), stack)
                n_ok += 1
            except Exception as e:
                print(f"skip {png.name}: {e}")
                n_skip += 1

    for ds, tmp, _ in nc_cache.values():
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    print(f"eddy_enh7 ready: npy_ok={n_ok} skip={n_skip} -> {dst}")


if __name__ == "__main__":
    main()
