# 涡旋模块：Faster R-CNN / Cascade R-CNN（框检测）与 YOLO-seg 对照说明

> **目的**：在**同一 polygon 标注**上做「二阶段检测（bbox）」对照，输出对比表或论文附图。  
> **必须与评委/读者说明**：主链 **YOLO 实例分割**用的是 **mask mAP@0.5（Ultralytics seg）**；本节 R-CNN 路线将多边形转为 **轴对齐 bbox**，指标为 **检测框 AP / mAP@0.5（torchvision 脚本内为 VOC 风格简化实现）**，二者**不能直接横向等同**，表中需**分列写明指标名称**。

---

## 1. 数据

### 1.1 生成 `data/processed/eddy`（仓库主链，须先有命题方 NetCDF）

命题方 `.nc` 放在 ``服创数据集/中尺度涡识别/``（与 `config/data.yaml` 的 `paths.raw_root` + `eddy_subdir` 一致）。由 OW 伪标签导出 YOLO-seg：

```bash
# 烟测（每 nc 少量帧）
python -m src.preprocess.eddy_dataset --export-yolo --data-config config/data.yaml ^
  --out data/processed/eddy --max-frames-per-file 3 --time-stride 60

# 正式（默认 time_stride=15；更密可 --time-stride 7）
python -m src.preprocess.eddy_dataset --export-yolo --data-config config/data.yaml ^
  --out data/processed/eddy --time-stride 7

python scripts/check_eddy_ready.py --dataset-yaml data/processed/eddy/dataset.yaml
```

实现：`src/preprocess/eddy_dataset.py` → `src/preprocess/eddy_yolo_export.py`。详见 `docs/架构与方法/涡旋_OW至YOLO伪标签开发参考.md`。

### 1.2 YOLO-seg → COCO Detection（bbox，供 R-CNN）

在 **1.1 已生成** `data/processed/eddy/dataset.yaml` 后：

```bash
python -m src.eddy.coco_bbox_export ^
  --dataset-yaml data/processed/eddy/dataset.yaml ^
  --out outputs/eddy_coco_bbox
```

**仅当**本机暂时没有命题方 `.nc`、又要验证脚本链路时，才用 `scripts/seed_rcnn_smoke_dataset.py` 生成占位数据（非论文用）。

- **`coco-root`**（训练时需传给 torchvision/MMDet）：与 **`dataset.yaml` 里的 `path`** 解析后的**数据集根目录**一致（JSON 里 `file_name` 相对该根）。
- **`outputs/eddy_coco_bbox/export_summary.json`** 中会写有 **`coco_root`**，可复制粘贴核对路径。
- 若希望导出目录自洽（图像拷贝进 `outputs/`）：加 **`--copy-images`**，此时 JSON 内路径会为 `images/{split}/文件名`。

实现：`src/eddy/coco_bbox_export.py`。导出曾因 **`img_id` 遗漏赋值**，新版本在每张云图上 **`img_id` 递增**，请先 **`pip install -r requirements.txt`** 后直接跑一次 export 校验 JSON。

---

## 2. Faster R-CNN（torchvision，仓库内脚本）

依赖：已有 **`torch` / `torchvision`**（见根目录 `requirements.txt`）。

```bash
python scripts/train_eddy_torchvision_detector.py ^
  --coco-root <上一步 export_summary 中的 coco_root> ^
  --train-json outputs/eddy_coco_bbox/train.json ^
  --val-json outputs/eddy_coco_bbox/val.json ^
  --epochs 12 --batch-size 4 --lr 0.005 ^
  --out outputs/eddy_detector_faster_rcnn
```

输出：

- `best.pt` / `last.pt`
- `summary.json`：`best_bbox_map50`（脚本内 **简化 VOC 风格** 的全集 AP 平均，**非 COCO API**）
- `train_log.txt`

**CUDA**：建议使用 GPU；CPU 仅可作冒烟。**极少数全无目标的样本**：torchvision 训练若报错可再在数据集侧过滤「零标注图像」（按需）。

---

## 3. Cascade R-CNN（MMDetection，需在环境中单独安装）

`torchvision` **不包含** Cascade R-CNN。推荐在项目虚拟环境中按需安装 **MMDetection**，并使用其与 **COCO** 对齐的评价（`bbox_mAP` 等）。

### 3.1 环境（示例）

```bash
pip install -U openmim
mim install mmengine mmcv mmdet
```

版本需彼此兼容，以 [OpenMMLab 安装页](https://mmdetection.readthedocs.io/en/latest/get_started.html) 为准。

### 3.2 数据布局

MMDet 常用写法：`data_root` + `ann_file` + `img_prefix`。若使用本仓库导出且**未** `--copy-images`，则：

- `data_root` = `dataset.yaml` 的 `path`（与 `--coco-root` 相同）
- `ann_file` = `outputs/eddy_coco_bbox/train.json`（或 val）

若使用了 **`--copy-images`**，则 `data_root` 可取 `outputs/eddy_coco_bbox`，`img_prefix` 与 JSON 内 `file_name` 前缀一致。

### 3.3 配置

在 MMDet 官方配置中检索 **Cascade R-CNN + R50 + FPN + COCO**，将 `num_classes` 改为 **2**（前景类数；配置里若写「类别数」需按 MMDet 版本说明区分是否含背景），并将 `data.train` / `data.val` 指向上面的 `ann_file` 与图像前缀。具体 config 片段因 MMDet 大版本而异，**不要硬抄旧版 YAML**，请以当前安装的 `mmdet` 自带 config 为模板复制修改。

训练与测试命令一般为：

```bash
mim train mmdet path/to/your_config.py
mim test mmdet path/to/your_config.py path/to/checkpoint.pth
```

将日志中的 **`bbox_mAP`**（及各类 AP）记入对比表。

---

## 4. 对比表模板（答辩/论文用）

| 方法 | 任务形式 | val 指标 | test 指标 | 备注 |
|------|----------|----------|-----------|------|
| YOLOv8-seg（主链） | 实例分割 | `mask_map50` | `mask_map50` | `src.eddy.eval` |
| Faster R-CNN | bbox 检测 | `bbox_*AP/mAP`（脚本简化口径） | 同上脚本改 val→test loader | `scripts/train_eddy_torchvision_detector.py` |
| Cascade R-CNN | bbox 检测 | `bbox_mAP`（MMDet） | MMDet test | 需单独安装 MMDet |

**制图建议**：并排 **原图 / GT 多边形 / YOLO mask / Faster 框 / Cascade 框**，图注写明「框由 polygon 外包矩形得到，用于检测对照」。

脚本（仓库内）：`python scripts/eddy_plot_method_compare_figure.py`（默认 val 前 3 张；第三、四列均为 **bbox** 画法对照 Faster R-CNN 与 YOLO；**不读图论证「为何选 YOLO」**，选用理由见正文与 ``mask_map50`` 等表）。

**GT 多边形 vs YOLO 分割（同一区域）**：`python scripts/eddy_plot_gt_vs_yolo_seg.py` → 默认 ``outputs/eddy/figures/gt_vs_yolo_seg.png``；``--seg-style contour`` 时 YOLO 侧仅轮廓线。

---

## 5. 相关路径

| 路径 | 说明 |
|------|------|
| `src/eddy/coco_bbox_export.py` | YOLO-seg → COCO bbox JSON |
| `scripts/train_eddy_torchvision_detector.py` | Faster R-CNN 训练与 val 简化评估 |
| `configs/mmdet/README.md` | Cascade / MMDet 占位说明 |
| `docs/实验与结果归档/涡旋模块工作汇总.md` | 主链 YOLO 口径与归档 |
