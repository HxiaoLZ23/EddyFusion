# MMDetection（Cascade R-CNN）与涡旋 COCO bbox 导出对接

本仓库 **不 pinned** `mmdet`/`mmcv`/`mmengine`（与赛题 PyTorch/xarray 主链分离），需在**单独虚拟环境**或使用 `openmim` 按需安装，避免与现有训练环境冲突。

## 与本项目导出衔接

1. 运行 `python -m src.eddy.coco_bbox_export --dataset-yaml ... --out outputs/eddy_coco_bbox`
2. 读取 `outputs/eddy_coco_bbox/export_summary.json` 中的 **`coco_root`**
3. 在 MMDet config 中将训练/验证 `ann_file` 指到 `train.json` / `val.json`，`img_prefix` 或 `data_root` 与 JSON 内 `file_name` 一致

若导出时使用 **`--copy-images`**，则常以 `outputs/eddy_coco_bbox` 为单一 `data_root`。

## Cascade R-CNN 配置从哪里来

不要用本目录下的虚构 YAML 糊弄训练。**请从你已安装的 MMDetection 包里复制官方 config**，例如（路径随版本变化）：

- `mmdet/.mim/configs/cascade_rcnn/` 下的 `cascade_*_fpn_*.py`

复制到本目录（例如 `configs/mmdet/eddy_cascade_rrcnn_r50_fpn_custom.py`）后修改：

- 数据集类型与 `ann_file`、`data_prefix` / `data_root`
- `num_classes = 2`（涡旋两类前景；若 config 注释要求「包含背景」需按该版本文档填写）

再执行：

```bash
mim train mmdet configs/mmdet/eddy_cascade_rrcnn_r50_fpn_custom.py
```

详细步骤见 `docs/实验与结果归档/涡旋_R-CNN与Cascade对比实验.md`。
