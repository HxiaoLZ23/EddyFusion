# GOAL 提示词：论文系统功能对齐（React + FastAPI）

> **用法**：整段复制到 Cursor `/goal`、新 Agent 会话首条消息，或配合 `.cursor/goal-paper-system.json` 逐 Phase 推进。  
> **上位文档**：`docs/开发规划/论文系统功能对齐与工程路线图.md`（v2，已审查：无水文、仅 React）。

---

## 一键粘贴版（Agent 首条）

```
【目标】按论文稿《格式-系统-修改版.docx》§4～§6，将仓库 React+FastAPI 系统对齐为可截图、可答辩、可写进论文的实现态。工程路线图见 docs/开发规划/论文系统功能对齐与工程路线图.md。

【范围】
- 做：web/、web_api/、src/ 中与预处理/涡旋/风浪/报告相关的 API 与页面。
- 不做：Streamlit app/（暂不管）、水文 src/hydro/ 与 HydroPanel（论文全文无水文）、7ch 主叙事与 UI（推理固定 3ch，E6 负结果仅附录）。

【论文口径】
- 前端：React；后端：FastAPI；两大智能模块：3ch YOLOv8-seg 涡旋 + 双头 LSTM 风浪异常 + DTW Top-K。
- 界面 §4.5：顶栏四入口（监测总览/涡旋分析/风浪分析/报告管理）；任务页左-中-右（上传+时间+ROI+任务类型 | 场图/曲线 | 统计+Top-K+报告导出）。
- 勿将「左涡旋+右上水文+右下风浪」三栏大屏当作论文界面。

【当前差距 P0】G01 去 7ch；G02 四入口导航；G03 时间+ROI+任务类型；G04/G05 变量映射与懒加载裁剪 API。

【执行规则】
1. 每轮只推进一个 Phase 或其中一个可验收子项；完成后更新 .cursor/goal-paper-system.json 的 phase/checklist。
2. 不破坏 AI_DEV_REQUIREMENTS：config level、run_eval_all、outputs/eddy_cloud_fair 权重与 mAP 口径。
3. 路径一律相对仓库根；不写死本机绝对路径。
4. 增依赖须同步 requirements.txt。
5. 每 Phase 结束在路线图 §3 差距表勾选对应 Gxx 状态。

【阶段顺序】Phase 0 信息架构 → Phase 1 预处理 API+左栏 → Phase 2 涡旋左中右 → Phase 3 风浪 LSTM+DTW+报告 → Phase 4 报告管理+演示脚本。

【本轮】从 Phase 0 开始：移除 DataSourceBar 7ch；DashboardLayout 改为论文四入口路由骨架；SHOW_HYDRO_UI 演示路径不展示。验收：/offline 仍可 3ch 双 MP4；顶栏路由与论文一致（离线/实时降为总览内数据源选项）。
```

---

## 任务定义

| 项 | 内容 |
|----|------|
| **Goal ID** | `paper-system-react-align` |
| **一句话** | 让 React 系统与论文 §4.5/§5 描述一致，支撑系统章截图与 §6 功能测试。 |
| **成功标准** | 离线 NC 一条链：上传→ROI/时间裁剪→涡旋场图+mask+统计→风浪预测曲线+异常等级+Top-K→结构化报告导出；顶栏与左-中-右布局可截图进论文。 |
| **非目标** | 水文模块、Streamlit 改版、7ch 推理入口、重训模型。 |

---

## Phase 清单（Agent 逐轮勾选）

### Phase 0 — 口径与信息架构（P0）

- [ ] `web/src/dashboard/DataSourceBar.tsx`：删除 7ch；`offlineSession` 默认且仅 `3ch`
- [ ] `web/src/layout/DashboardLayout.tsx`：顶栏 → 监测总览 / 涡旋分析 / 风浪分析 / 报告管理
- [ ] 路由：`/monitor`、`/eddy`、`/windwave`、`/reports`；`/offline`、`/realtime` 并入总览数据源子模式
- [ ] 论文演示路径不渲染 Hydro（`SHOW_HYDRO_UI` 对答辩路由无效或恒 false）
- [ ] OpenAPI/注释不出现「水文为论文模块」表述

**验收**：3ch 双 MP4 可生成；浏览器顶栏与论文四入口一致。

### Phase 1 — 预处理 + 左栏配置（P0）

- [ ] `config/nc_variable_map.yaml`（对齐论文表 5-2）
- [ ] `src/preprocess/nc_lazy_subset.py` + Facade 真实 `open_lazy`/`sel`
- [ ] `web_api/routers/preprocess.py`：`POST /api/preprocess/subset`
- [ ] `web/src/.../TaskConfigPanel.tsx`：上传、time_start/end、bbox、任务类型

**验收**：小 NC 烟测裁剪 API；左栏表单可提交并拿到 meta JSON。

### Phase 2 — 涡旋任务页左-中-右（P1）

- [ ] `POST /api/eddy/preview` 或 `eddy/frame`：ADT 栅格 + u/v 矢量 + mask + stats
- [ ] `EddyMapView`、`EddyStatsPanel`；时间轴 `time_index`
- [ ] 双 MP4 保留为动画子模式；异步 job + 进度

**验收**：涡旋页可截图（场图+mask+右栏统计表）。

### Phase 3 — 风浪 LSTM + 异常 + DTW（P1）

- [ ] `POST /api/windwave/forecast`：双头 LSTM 曲线 + anomaly_segments + levels
- [ ] 风浪页中栏曲线+异常 shading；右栏黄橙红 + Top-K（少跳转 TyphoonKbPage）
- [ ] `POST /api/report/structured`：§5.5.5 字段 → MD/PDF

**验收**：预测→异常→Top-K→报告一条链可演示。

### Phase 4 — 报告管理与收尾（P2）

- [ ] `/reports` 历史任务列表与再导出
- [ ] `docs/开发规划/答辩演示脚本.md`（仅 React）
- [ ] 路线图 §3 G01–G16 状态列更新

**验收**：§6.3 功能测试表所列用例均可勾选。

---

## 关键文件索引

| 用途 | 路径 |
|------|------|
| 路线图 | `docs/开发规划/论文系统功能对齐与工程路线图.md` |
| 阶段状态 | `.cursor/goal-paper-system.json` |
| React 壳 | `web/src/layout/DashboardLayout.tsx`，`web/src/App.tsx` |
| 旧大屏（待重构） | `web/src/dashboard/OceanDashboard.tsx` |
| 涡旋 API | `web_api/routers/eddy_dual.py` |
| 风浪 API | `web_api/routers/windwave_report.py` |
| 台风 KB | `web_api/routers/typhoon_kb.py` |
| 3ch 权重 | `outputs/eddy_v6_b0_fair/best.pt`（`config/eddy.yaml`） |
| 不可改规范 | `相关文件/AI_DEV_REQUIREMENTS.md` |

---

## 禁止事项（每轮自检）

1. 不要把水文/ConvLSTM 写进论文对齐 PR 或截图说明。  
2. 不要恢复或强化 7ch UI/文案作为默认方案。  
3. 不要以 Streamlit 改动作为本 Goal 的完成项。  
4. 不要破坏 `run_eval_all` 与 eddy 评测指标链。  
5. 不要把「三栏涡旋|水文|风浪」当作论文验收布局。

---

## 轮次结束汇报模板

```markdown
## Goal 轮次汇报
- Phase：Phase X / 子项名称
- 完成：…
- 变更文件：…
- 验收：通过 / 部分 / 未测
- goal-paper-system.json：phase=…，checklist 已勾选 …
- 下一步：Phase X+1 第一项 …
- 阻塞：无 / …
```
