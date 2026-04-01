# EddyFusion

频域先验与多时间尺度驱动的海洋智能分析平台（赛题 A09）。本仓库为可运行代码与配置；**全局开发规范**见本地或团队共享的 `相关文件/AI_DEV_REQUIREMENTS.md`

## 环境

- Python 3.9+（推荐 3.10；当前若使用 3.12 需自行确认与命题方环境一致）
- PyTorch **≥2.5.1**（按 [pytorch.org](https://pytorch.org) 选择与 CUDA 匹配的 wheel；`pip` 默认常为 **CPU** 版，训练请按需改装 GPU 版）
- Linux / WSL2 推荐用于训练与推理

### 虚拟环境（仓库根目录执行）

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

命题方原始数据目录在 `config/data.yaml` 的 `paths.raw_root`（默认 **`服创数据集/`**），预处理脚本会递归扫描其下 `*.nc`。

在仓库根目录执行 `python -m ...`（Windows / Linux 均如此）。

## 目录结构

与《A09-项目开发文档》一致：`config/`、`data/`、`src/`、`scripts/`、`outputs/`、`docs/`、`submission/`。

## 阶段一：无命题方数据时的烟测（合成 / 内置小集）

| 模块 | 命令 |
|------|------|
| 水文 ConvLSTM | `python -m src.hydro.train --synthetic` → `python -m src.hydro.eval --config config/hydro_synthetic.yaml --ckpt outputs/hydro/best.pt` |
| 风-浪 LSTM | `python -m src.anomaly.train --synthetic` → `python -m src.anomaly.eval` |
| 涡旋 YOLO-seg | `python -m src.eddy.train --smoke`（使用 ultralytics 内置 `coco8-seg`，需联网下载） |
| NetCDF 检查 | 将 `*.nc` 放入 `data/raw/` 后：`python -m src.preprocess.netcdf_io --config config/data.yaml` |

全量数据就绪后：预处理 → 各模块 `train` / `eval`，详见 `scripts/*.sh`。

## 运行顺序（数据就绪后）

1. 预处理：`bash scripts/run_preprocess.sh`（或分步执行 `python -m src.preprocess.*`）
2. 训练：`bash scripts/run_eddy_train.sh` 等
3. 评估：`bash scripts/run_eval_all.sh`（需各模块已产出 `outputs/*/best.pt`）
4. 演示：`python -m src.demo.app_gradio`（实现后）

## 远程仓库

```bash
git remote add origin git@github.com:HxiaoLZ23/EddyFusion.git
git push -u origin main
```

## 数据与权重

`data/raw/`、`data/processed/` 下大文件与 `outputs/` 权重默认不提交；见 `.gitignore`。
