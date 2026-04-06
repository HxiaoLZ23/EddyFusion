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

**Windows + 中文路径**：若资源管理器能打开 `.nc` 但 `netcdf_io` 报错，脚本已自动尝试短路径与临时英文副本；请在 **`F:\创赛`**（仓库根）下执行命令，或使用正斜杠：`--path 服创数据集/海域要素预测/1994/19940101.nc`。

在仓库根目录执行 `python -m ...`（Windows / Linux 均如此）。

### 云服务器（AutoDL 等）

- 项目目录名：**EddyFusion**（注意大小写；Linux 下 `Eddyfusion` 与 `EddyFusion` 为不同路径）
- 典型克隆/工作路径：`~/autodl-tmp/EddyFusion`（root 用户下为 `/root/autodl-tmp/EddyFusion`）
- 下文与脚本中的「仓库根」在云服务器上即指上述路径。
- **命题方数据位置二选一即可**（`config/data.yaml` 默认 `paths.raw_root: "服创数据集"`，相对仓库根解析）：
  - **放在仓库内**：`~/autodl-tmp/EddyFusion/服创数据集/`（与 `config/` 同级），则**无需**运行 `setup_fuchuang_to_data.sh`，预处理会直接读该目录。
  - **放在仓库外**（如 `~/autodl-tmp/服创数据集`，避免占满系统盘）：用 `scripts/setup_fuchuang_to_data.sh` 的 `FU_MODE=link` 在仓库根创建软链 `服创数据集` → 实际数据目录。

### 全流程命令（云服务器 · 命题方 NetCDF → 水文训练）

以下均在仓库根执行（如 `cd ~/autodl-tmp/EddyFusion`），且已创建并激活 `.venv`、`pip install -r requirements.txt`。

**1. 放入命题方数据**  
将「服创数据集」解压/上传，使目录结构为 **`服创数据集/海域要素预测/.../*.nc`**（或命题方提供的其它子目录名，与 `config/data.yaml` 的 `hydro_subdir` 一致）。

- **方式 A（推荐简单）**：直接放在 **EddyFusion 仓库根下**，即  
  `~/autodl-tmp/EddyFusion/服创数据集/`  
  与 `config/`、`src/` 同级。完成后**跳过**下面步骤 2。

- **方式 B**：放在仓库外，例如 `~/autodl-tmp/服创数据集`（其下含 `海域要素预测/`）。**勿在系统盘空间不足时再向 `/data` 全量复制**；然后执行步骤 2 做软链。

**2. 接到项目（仅方式 B：零拷贝软链）**

```bash
export FU_MODE=link
export FU_CHUANG_SRC="$HOME/autodl-tmp/服创数据集"
bash scripts/setup_fuchuang_to_data.sh
```

完成后仓库根会出现 `服创数据集` → 指向 `$FU_CHUANG_SRC`，`config/data.yaml` 的 `paths.raw_root` 无需改。

**3. 预处理（生成 `data/processed/hydro/*.npz`）**

```bash
# 试跑：少量日文件；滑窗步长勿长期用 1（会生成巨量窗口、内存可达百 GB 级似卡死），建议 12～24
python -m src.preprocess.hydro_dataset \
  --config config/hydro_hycom.yaml \
  --from-nc --data-config config/data.yaml \
  --max-daily-files 120 --stride 24

# 全量：先在 config/data.yaml 将 hydro_preprocess.max_daily_files 设为 null（或删去限制），再：
# python -m src.preprocess.hydro_dataset --config config/hydro_hycom.yaml --from-nc --data-config config/data.yaml --stride 1

# 按命题方年份划分训练/验证/测试时，先启用 data.yaml 的 hydro_year_split.enabled，再：
# python -m src.preprocess.hydro_dataset --config config/hydro_hycom.yaml --from-nc --data-config config/data.yaml --year-split --stride 1
```

**4. 训练**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 可选，缓解显存碎片
python -m src.hydro.train --config config/hydro_hycom.yaml
```

**5. 评估**

```bash
python -m src.hydro.eval --config config/hydro_hycom.yaml --ckpt outputs/hydro/best.pt
# 若训练过程出现 nan 未写出 best.pt，可改用: --ckpt outputs/hydro/last.pt
```

说明：单日日文件拼接后的总时间步须 ≥ `input_steps + output_steps`（见 `config/hydro_hycom.yaml`）；网格大时请用 GPU。

## 目录结构

与《A09-项目开发文档》一致：`config/`、`data/`、`src/`、`scripts/`、`outputs/`、`docs/`、`submission/`。命题方原始数据可置于仓库根下 **`服创数据集/`**（默认不提交，见 `.gitignore`）。

## 阶段一：无命题方数据时的烟测（合成 / 内置小集）

| 模块 | 命令 |
|------|------|
| 水文 ConvLSTM | `python -m src.hydro.train --synthetic` → `python -m src.hydro.eval --config config/hydro_synthetic.yaml --ckpt outputs/hydro/best.pt` |
| 风-浪 LSTM | `python -m src.anomaly.train --synthetic` → `python -m src.anomaly.eval` |
| 涡旋 YOLO-seg | `python -m src.eddy.train --smoke`（使用 ultralytics 内置 `coco8-seg`，需联网下载） |
| NetCDF 检查 | 将 `*.nc` 放入 `data/raw/` 后：`python -m src.preprocess.netcdf_io --config config/data.yaml` |

全量数据就绪后：预处理 → 各模块 `train` / `eval`，详见 `scripts/*.sh`。

### 命题方海域要素预测（HYCOM NetCDF → 水文 npz）

需已配置 `config/data.yaml` 中 `paths.raw_root`（默认 `服创数据集`）与 `hydro_subdir`（默认 `海域要素预测`）。**单文件 time 步数可能因日而异**，拼接后总长度须 ≥ `input_steps + output_steps`（`hydro_hycom.yaml` 默认为 140）。

```powershell
# 示例：先用 20 个日文件、滑窗步长 24，生成 data/processed/hydro/*.npz
python -m src.preprocess.hydro_dataset --config config/hydro_hycom.yaml --from-nc --data-config config/data.yaml --max-daily-files 20 --stride 24

python -m src.hydro.train --config config/hydro_hycom.yaml
python -m src.hydro.eval --config config/hydro_hycom.yaml --ckpt outputs/hydro/best.pt
```

全量处理时可在 `config/data.yaml` 的 `hydro_preprocess.max_daily_files` 设为 `null`，并酌情调大 `window_stride` 控制样本量。138×125 网格在 **CPU** 上训练极慢，建议使用 **GPU**。

## 运行顺序（数据就绪后）

1. 预处理：`bash scripts/run_preprocess.sh`（或分步执行 `python -m src.preprocess.*`）
2. 训练：`bash scripts/run_eddy_train.sh` 等
3. 评估：`bash scripts/run_eval_all.sh`（需各模块已产出 `outputs/*/best.pt`）
4. 演示：`python -m src.demo.app_gradio`  
   - **界面开发说明**（功能清单、Gradio/Streamlit、验收标准、与配置衔接）见 **`相关文件/A09-项目开发文档.md` 第 5.4 节**。  
   - 当前若入口仍为占位，需按该节完成 P0 后再用于正式录屏；无全量数据时可先用合成/烟测与占位流程保证可启动。

## 远程仓库

```bash
git remote add origin git@github.com:HxiaoLZ23/EddyFusion.git
git push -u origin main
```

## 数据与权重

`data/raw/`、`data/processed/` 下大文件与 `outputs/` 权重默认不提交；见 `.gitignore`。
