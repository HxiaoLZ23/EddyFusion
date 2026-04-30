from __future__ import annotations

import argparse
from pathlib import Path

import yaml


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_COMMON_PT_NAMES = (
    "yolov8n-seg.pt",
    "yolov8n.pt",
    "yolov8s-seg.pt",
    "yolo11n.pt",
    "yolo11n-seg.pt",
)


def _read_dataset_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"未找到 dataset yaml: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"dataset yaml 不是字典结构: {path}")
    return data


def _resolve_dataset_path(dataset_yaml: Path, data: dict) -> Path:
    base = data.get("path", ".")
    p = Path(base)
    if p.is_absolute():
        return p
    # Ultralytics 常见写法是 path: data/processed/eddy（相对仓库根），
    # 若这里盲目按 dataset.yaml 同目录拼接会得到 .../data/processed/eddy/data/processed/eddy。
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (dataset_yaml.parent / p).resolve()


def _count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    n = 0
    for ext in _IMG_EXTS:
        n += len(list(d.glob(f"*{ext}")))
    return n


def _count_labels(d: Path) -> int:
    if not d.is_dir():
        return 0
    return len(list(d.glob("*.txt")))


def _check_split(root: Path, split_rel: str, *, strict: bool) -> bool:
    img_dir = (root / split_rel).resolve()
    label_rel = split_rel.replace("images/", "labels/")
    label_dir = (root / label_rel).resolve()
    n_img = _count_images(img_dir)
    n_lab = _count_labels(label_dir)
    ok = True
    print(f"[split] {split_rel}: images={n_img} labels={n_lab}")
    if not img_dir.is_dir():
        print(f"  - 缺目录: {img_dir}")
        ok = False
    if not label_dir.is_dir():
        print(f"  - 缺目录: {label_dir}")
        ok = False
    if n_img == 0:
        print("  - 图像数为 0")
        ok = False
    if n_lab == 0:
        print("  - 标签数为 0")
        ok = False
    if n_lab > 0 and n_img > 0 and n_lab < int(0.5 * n_img):
        print("  - 标签远少于图像（<50%），可能阈值过严或导出异常")
        if strict:
            ok = False
    return ok


def _check_corrupt_weights() -> list[Path]:
    bad: list[Path] = []
    try:
        import torch
    except Exception:
        print("[weights] 未安装 torch，跳过损坏权重检查")
        return bad
    roots = [
        Path.cwd(),
        Path.home() / ".cache" / "ultralytics",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        Path.home() / ".config" / "Ultralytics",
    ]
    for name in _COMMON_PT_NAMES:
        for r in roots:
            p = r / name
            if not p.is_file():
                continue
            try:
                try:
                    torch.load(p, map_location="cpu", weights_only=True)
                except TypeError:
                    torch.load(p, map_location="cpu")
            except Exception:
                bad.append(p)
    if bad:
        print("[weights] 检测到疑似损坏权重：")
        for p in bad:
            print(f"  - {p}")
    else:
        print("[weights] 常见预训练权重读取正常")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="eddy 训练前体检：dataset/样本/损坏权重")
    parser.add_argument("--dataset-yaml", type=str, default="data/processed/eddy/dataset.yaml")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：标签少于图像 50% 时也视为失败",
    )
    args = parser.parse_args()

    ds_yaml = Path(args.dataset_yaml).resolve()
    data = _read_dataset_yaml(ds_yaml)
    root = _resolve_dataset_path(ds_yaml, data)
    train_rel = str(data.get("train", "images/train"))
    val_rel = str(data.get("val", "images/val"))
    print(f"[dataset] yaml={ds_yaml}")
    print(f"[dataset] root={root}")
    ok_train = _check_split(root, train_rel, strict=bool(args.strict))
    ok_val = _check_split(root, val_rel, strict=bool(args.strict))
    bad = _check_corrupt_weights()
    ok = ok_train and ok_val and (len(bad) == 0)
    print(f"[result] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
