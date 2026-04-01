# EddyFusion

频域先验与多时间尺度驱动的海洋智能分析平台（赛题 A09）。本仓库为可运行代码与配置；**全局开发规范**见本地或团队共享的 `相关文件/AI_DEV_REQUIREMENTS.md`

## 环境

- Python 3.9+（推荐 3.10）
- PyTorch **≥2.5.1**（按 [pytorch.org](https://pytorch.org) 选择与 CUDA 匹配的 wheel）
- Linux / WSL2 推荐用于训练与推理

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

验证：`python -c "import torch; print(torch.__version__)"`

## 目录结构

与《A09-项目开发文档》一致：`config/`、`data/`、`src/`、`scripts/`、`outputs/`、`docs/`、`submission/`。

## 运行顺序（数据就绪后）

1. 预处理：`bash scripts/run_preprocess.sh`（或分步执行 `python -m src.preprocess.*`）
2. 训练：`bash scripts/run_eddy_train.sh` 等
3. 评估：`bash scripts/run_eval_all.sh`
4. 演示：`python -m src.demo.app_gradio`（实现后）

当前各模块入口为**占位实现**，需按《A09-分阶段详细执行指南》阶段一继续开发。

## 远程仓库

```bash
git remote add origin git@github.com:HxiaoLZ23/EddyFusion.git
git push -u origin main
```

## 数据与权重

`data/raw/`、`data/processed/` 下大文件与 `outputs/` 权重默认不提交；见 `.gitignore`。
