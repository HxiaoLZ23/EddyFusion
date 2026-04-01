from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import project_root, resolve_path


def write_template_dataset_yaml(out_path: Path, root: Path) -> None:
    """生成 ultralytics 用 dataset 模板；图像与标注需人工放入对应目录。"""
    # 使用正斜杠便于 YOLO 在 Windows/Linux 下读取
    rel = Path("data/processed/eddy/images")
    txt = f"""# 将训练/验证图像与标注按 YOLO-seg 格式放入 {rel}/train 与 {rel}/val
path: {rel.as_posix()}
train: train/images
val: val/images
names:
  0: eddy
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8")
    print(f"已写入模板: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/data.yaml")
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="在 config/eddy_dataset_template.yaml 写入 YOLO 数据说明模板",
    )
    args = parser.parse_args()
    root = project_root()
    if args.write_template:
        write_template_dataset_yaml(root / "config" / "eddy_dataset_template.yaml", root)
        return
    raise NotImplementedError(
        "命题方涡旋图像与标注就绪后实现解析；可先 --write-template 生成数据集说明模板"
    )


if __name__ == "__main__":
    main()
