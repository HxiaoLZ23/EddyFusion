from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_yaml, project_root, resolve_path


def write_template_dataset_yaml(out_path: Path, root: Path) -> None:
    """生成 ultralytics 用 dataset 模板；图像与标注需人工放入对应目录。"""
    rel = Path("data/processed/eddy")
    txt = f"""# 将训练/验证图像与标注按 YOLO-seg 格式放入 {rel}/images/{'{train,val}'}
path: {rel.as_posix()}
train: images/train
val: images/val
names:
  0: eddy_cyclonic
  1: eddy_anticyclonic
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8")
    print(f"已写入模板: {out_path}")


def _raw_eddy_dir(data_cfg: dict) -> Path:
    paths = data_cfg.get("paths", {})
    raw = resolve_path(paths.get("raw_root", "服创数据集"))
    sub = paths.get("eddy_subdir", "中尺度涡识别")
    return raw / sub


def inspect_eddy_netcdf(nc_path: Path) -> None:
    try:
        import xarray as xr
    except ImportError as e:
        raise SystemExit("需要安装 xarray（见 requirements.txt）才能 --inspect") from e
    ds = xr.open_dataset(nc_path)
    print(f"文件: {nc_path}")
    print("dims:", dict(ds.sizes))
    print("data_vars:", list(ds.data_vars))
    print("coords:", list(ds.coords))
    for name in list(ds.data_vars)[:25]:
        v = ds[name]
        print(f"  {name}: dims={v.dims} shape={v.shape} dtype={v.dtype}")
    ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="涡旋数据：YOLO 模板 / NetCDF 结构探查 / OW 伪标签导出")
    parser.add_argument("--config", type=str, default="config/data.yaml")
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="在 config/eddy_dataset_template.yaml 写入 YOLO 数据说明模板",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="读取命题方涡旋目录下首个 .nc 并打印维度与变量（需 xarray）",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="指定单个 NetCDF 路径（可选；默认取涡旋子目录下字典序第一个 .nc）",
    )
    parser.add_argument(
        "--export-yolo",
        action="store_true",
        help="从 adt/ugos/vgos 计算 OW（多分位投票）并导出 YOLO-seg 用 PNG+txt；其余参数见 eddy_yolo_export",
    )
    args, rest = parser.parse_known_args()
    root = project_root()
    if args.export_yolo:
        from src.preprocess.eddy_yolo_export import main_argv

        raise SystemExit(main_argv(rest))
    if rest:
        parser.error(f"未识别的参数（若导出伪标签请加 --export-yolo）: {rest}")
    if args.write_template:
        write_template_dataset_yaml(root / "config" / "eddy_dataset_template.yaml", root)
        return
    if args.inspect:
        data_cfg = load_yaml(root / args.config)
        if args.path:
            p = resolve_path(args.path)
        else:
            eddy_dir = _raw_eddy_dir(data_cfg)
            if not eddy_dir.is_dir():
                raise FileNotFoundError(f"未找到涡旋目录: {eddy_dir}")
            ncs = sorted(eddy_dir.glob("*.nc"))
            if not ncs:
                raise FileNotFoundError(f"目录下无 .nc: {eddy_dir}")
            p = ncs[0]
        inspect_eddy_netcdf(p)
        return
    raise SystemExit(
        "请使用: --write-template | --inspect | --export-yolo（OW 伪标签 → YOLO-seg，详见 docs/涡旋_OW至YOLO伪标签开发参考.md）"
    )


if __name__ == "__main__":
    main()
