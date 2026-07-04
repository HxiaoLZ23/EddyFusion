# EddyFusion — 海洋 AI 三模块系统

> **论文对应版本**：本仓库即为毕业论文《基于深度学习的海洋涡旋识别与风浪预警系统设计》（黄柏霖，软件222）终稿所述实现。  
> 论文实现线：**React + FastAPI**，涡旋（OW→YOLOv8-seg）+ 风浪（双头 LSTM + 3σ + DTW）；详见 [`submission/thesis/README.md`](submission/thesis/README.md)。

面向 A09 赛题的**涡旋检测 · 水文预测 · 风浪异常**一体化工程：算法训练/评估、FastAPI 推理服务、React **论文演示系统**与 Streamlit 早期原型。

| 模块 | 方法概要 | 论文 | 主要入口 |
|------|----------|:----:|----------|
| **涡旋** | OW 伪标签 → YOLOv8-seg（在线 3ch） | ✅ | `src/eddy/`、`config/eddy*.yaml` |
| **风浪异常** | 共享 LSTM 双头一步预测 + 残差 3σ + DTW 弱关联 | ✅ | `src/anomaly/` |
| **水文** | ConvLSTM 多步预测 + 物理尺度 NRMSE | 赛题扩展 | `src/hydro/`、`config/hydro*.yaml` |
---

## 环境

- **Python ≥ 3.10**，**PyTorch ≥ 2.5.1**（见 `requirements.txt`）
- 可选：**Node.js 18+**（React 前端）、**CUDA**（训练/GPU 物理场）
- 数据与权重**不入库**：本地放置于 `data/`、`outputs/`（见 `.gitignore`）

```powershell
cd F:\创赛   # 或你的克隆路径
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 快速启动（论文演示系统）

**论文 §5.5 验收路径**：React + FastAPI 双服务（NetCDF 上传与裁剪、涡旋/风浪分析、报告导出）。

顶栏四入口与论文 §4.5 一致：**监测总览 · 涡旋分析 · 风浪分析 · 报告管理**（`web/src/layout/DashboardLayout.tsx`）。
```powershell
# 终端 1 — API（默认 http://127.0.0.1:8000）
.\scripts\run_web_api.ps1

# 终端 2 — 前端（默认 http://127.0.0.1:5173）
cd web
npm install
npm run dev
```

- 前端说明：`web/README.md`
- API 路由与离线会话：`web_api/README.md`
- Streamlit 旧版/离线页：`app/README.md`（`streamlit run app/main.py`）

功能开关见 `web/src/featureFlags.ts`、`app/feature_flags.py`。

---

## 仓库结构

```
├── config/          # 三模块 YAML（eddy / hydro / anomaly / data）
├── src/
│   ├── eddy/        # 训练、评估、NC→BGR、物理 8 通道、MP4 导出
│   ├── hydro/       # ConvLSTM 训练与扩展指标
│   ├── anomaly/     # 检测、LLM 报告、台风桥接
│   └── preprocess/  # 数据集构建、YOLO 导出、NC 懒加载
├── scripts/         # 训练、评估、消融、归档脚本
├── web/             # React + Vite 论文系统
├── web_api/         # FastAPI（/api/*）
├── app/             # Streamlit 演示与离线系统
├── docs/            # 工程手册、实验归档、开发规划（见索引）
├── submission/      # 论文用表格、图、答辩材料
├── tests/           # pytest（异常检测、涡旋物理、API）
└── outputs/         # 本地训练/推理产物（gitignore）
```

**文档入口**：[`docs/文档分类索引.md`](docs/文档分类索引.md)

---

## 训练与评估（概要）

路径均相对**仓库根**；云端训练日志为产物溯源依据。

| 模块 | 典型命令 |
|------|----------|
| 涡旋 YOLO | `python -m src.eddy.train --config config/eddy_enh7.yaml` |
| 涡旋评估 | `python -m src.eddy.eval --weights outputs/.../best.pt` |
| 水文 | `python -m src.hydro.train --config config/hydro_hycom.yaml` |
| 风浪 | `python -m src.anomaly.detect` / `eval`（见 `config/data.yaml`） |

消融与云端脚本：`scripts/run_eddy_*`、`scripts/hydro_cloud_assessment.py` 等。  
指标与实验归档：`docs/实验与结果归档/`、`submission/tables/`。

---

## 数据与配置

- 数据说明：[`data/README_data.md`](data/README_data.md)
- 变量映射：`config/nc_variable_map.yaml`
- LLM（百炼）示例：`config/dashscope.local.json.example` → 本地 `config/dashscope.local.json`（已 gitignore）

**不可擅自更改**（交接/评测口径）：config 中 `level` 含义、eval 输出字段、数据层路径与 tensor 形状、PyTorch 版本下限。详见 `相关文件/AI_DEV_REQUIREMENTS.md`。

---

## 测试

```powershell
pytest tests/ -q
python scripts/run_system_tests.py   # 可选：UI/API 归档
```

---

## 提交材料

- 论文镜像与代码对应说明：`submission/thesis/README.md`
- 表格/图：`submission/tables/`、`submission/figures/`
- 答辩 Q&A：`submission/答辩_前后端常见问题.md`
- 云端结果说明：`outputs/cloud/README.md`
---

## 许可与引用

赛题与命题方数据使用须遵守赛事规定；公开权重与大数据集请勿提交至 Git（见 `.gitignore`）。
