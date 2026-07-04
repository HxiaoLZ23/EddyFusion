# GOAL 提示词：定稿论文 docx→md 降 AIGC（最小语义改动）

> **用法**：整段复制到 Cursor `/goal`、新 Agent 会话首条消息，或配合 `.cursor/goal-thesis-de-aigc.json` 逐 Phase 推进。  
> **定稿源文件（只读参照，勿覆盖）**：`c:\Users\HxiaoL\Desktop\系统模式论文\基于深度学习的海洋涡旋识别与风浪预警系统设计.docx`  
> **仓库内技术口径参照**：`submission/_thesis_v6_review.md`、`submission/_thesis_dtw_regional_mean_vs_center_draft.md`、`docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md`

---

## 一键粘贴版（Agent 首条）

```
【目标】将定稿论文《基于深度学习的海洋涡旋识别与风浪预警系统设计.docx》转为 Markdown，在**不改变技术事实、数据、公式、图表编号与结论**的前提下降低 AIGC 检出率；并另写《降 AIGC 修订对照》逐处说明改动。

【定稿路径（源，勿删改）】
c:\Users\HxiaoL\Desktop\系统模式论文\基于深度学习的海洋涡旋识别与风浪预警系统设计.docx

【产出路径（仓库内）】
1. submission/thesis/基于深度学习的海洋涡旋识别与风浪预警系统设计_原文镜像.md   — docx 忠实转 md（本轮尽量不改写，作 diff 基线）
2. submission/thesis/基于深度学习的海洋涡旋识别与风浪预警系统设计_降AIGC.md — 降 AIGC 工作稿（唯一允许改写的正文）
3. submission/thesis/降AIGC修订对照.md — 按章/节列出：原文摘录 → 改后摘录 → 改动类型 → 是否动及技术含义

【硬性约束 — 禁止改动的内容】
- 所有数值、表格单元格、实验 ID、mAP/MAE/Recall、样本量、Top-K、τ=1.5 等参数与表号图号
- 公式符号含义与编号（可与 v6 公式审计对齐，但不得为「润色」改公式）
- 参考文献条目、图表题注中的专名（YOLOv8-seg、IBTrACS、NetCDF、LSTM 等）
- 章节标题层级与学校格式要求的结构（章节目录树与定稿一致）

【允许且鼓励的降 AIGC 手段（对原文影响最小）】
- 句式：拆长句、合并重复短句；主被动交替；减少「首先/其次/再次/综上所述/值得注意的是/随着…的发展」等模板连接
- 词汇：同义替换非术语词；减少排比三连；避免段首雷同
- 信息：用本仓库**已实现细节**替换空泛套话（如具体 API 名、config 键、窗口 pad=4 clamp 等），增加可核查具体性
- 冗余：删与前后段完全重复的总结句（删前在对照表登记）
- 禁止：为降率而编造实验、改结论、删 limitation、把 DTW 写成参与 3σ 判定

【转换规则（docx→md）】
- 优先 pandoc：`pandoc -f docx -t gfm --wrap=none`；失败则用 python-docx 按段落顺序导出
- 图片：提取到 submission/thesis/assets/，md 内相对路径引用
- 表格：保留为 GFM 表格或 HTML 表，**单元格文字与 docx 一致**
- 转 md 阶段不做降 AIGC，先产出「原文镜像」再复制为工作稿

【降 AIGC 执行顺序】
Phase 0：转 md + 基线镜像 + 章节目录与字数统计
Phase 1：摘要、绪论、相关技术（套话最密，优先）
Phase 2：系统设计章（§4）— 只改表述，接口/模块名与代码一致
Phase 3：算法章（§5）— 严格对照 detect.py / model.py / eddy 口径，公式不动
Phase 4：实验章（§6）— 表注可补一句口径说明，表体数字不动
Phase 5：总结与展望（§7）— limitation 保留并写清
Phase 6：全文通读 + 对照表补全 + 可选 pandoc 回 docx（另存，不覆盖桌面定稿）

【对照表格式（降AIGC修订对照.md）】
每条必须含：`| 章.节 | 改动类型 | 原文(≤80字) | 改后(≤80字) | 技术含义是否变化 |`
改动类型枚举：套话删减 / 句式重组 / 同义替换 / 具体化(仓库事实) / 合并重复 / 标点格式

【验收标准】
- [ ] 原文镜像 md 与 docx 段落一一对应，无整节丢失
- [ ] 降 AIGC md 中所有表图编号、数值与定稿一致（抽样 + 表 6-2/6-3 全量核对）
- [ ] 修订对照 ≥ 80% 改动条目有登记（允许「仅标点」批量一条说明）
- [ ] v6 review 中已标 P0 的 DTW/anomaly_index 口径在降 AIGC 稿中未被改回旧表述
- [ ] 不将定稿 docx 覆盖写回桌面路径；若导出 docx 仅放 submission/thesis/

【本轮】从 Phase 0 开始：确认定稿 docx 可读 → 创建 submission/thesis/ → 产出原文镜像 md 与目录统计；不开始大规模改写直至镜像 diff 可核对。
```

---

## 任务定义

| 项 | 内容 |
|----|------|
| **Goal ID** | `thesis-de-aigc` |
| **一句话** | 定稿论文转 MD 后，以最小技术语义变动降低 AIGC 率，并留可追溯修订对照。 |
| **成功标准** | 三份交付物齐全；数值/公式/结论与定稿一致；对照表可答辩追溯；用户可自选查重/AIGC 平台复测。 |
| **非目标** | 重做大纲、增删章节、重跑实验改表、替用户承诺具体 AIGC 百分比、覆盖桌面定稿 docx。 |

---

## Phase 清单（Agent 逐轮勾选）

### Phase 0 — docx→md 基线（不改写）

- [ ] 确认源文件存在且可读（~4.4MB docx）
- [ ] 创建 `submission/thesis/`、`submission/thesis/assets/`
- [ ] 导出 `…_原文镜像.md`（pandoc 或 python-docx）
- [ ] 提取图片至 `assets/`，检查图题与引用
- [ ] 生成 `章节目录与字数统计.md`（章/节行数、总字数）
- [ ] 复制镜像为 `…_降AIGC.md` 工作副本

**验收**：镜像 md 章节目录与 docx 一致；任意一节可人工 spot-check 3 段无丢字。

### Phase 1 — 前置章节降 AIGC

- [ ] 摘要（中英文若存在）
- [ ] 第 1 章 绪论
- [ ] 第 2 章 相关技术/理论基础
- [ ] 本章改动写入 `降AIGC修订对照.md`

**验收**：无新增未实现功能描述；无改动研究结论方向。

### Phase 2 — 系统设计章（§4）

- [ ] 总体架构、模块划分、界面描述与 `web/`、`web_api/` 一致
- [ ] 不写水文/ConvLSTM 为论文主模块（与 `GOAL_论文系统React对齐.md` 一致）
- [ ] 登记对照表

### Phase 3 — 算法章（§5）

- [ ] 涡旋：3ch + YOLOv8-seg；7ch 仅消融表述
- [ ] 风浪：`WindWaveLSTM`、`anomaly_index`（非有量纲 current_curve 作主指标）、异常窗 + DTW 弱关联
- [ ] 公式与 `submission/_thesis_v6_formulas_audit.md` 交叉核对
- [ ] 登记对照表

### Phase 4 — 实验章（§6）

- [ ] 表体数字不动；表注/正文可补口径脚注
- [ ] 表 6-3、台风 Top-K、DTW 描述与 `docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md` 一致
- [ ] 登记对照表

### Phase 5 — 总结与展望（§7）

- [ ] limitation 保留（DTW 尺度、Oracle 评测、区域平均风等）
- [ ] 避免空洞「未来工作」排比
- [ ] 登记对照表

### Phase 6 — 收尾

- [ ] 全文搜索并消灭降 AIGC 稿中的高风险套话残留（见下「套话黑名单」）
- [ ] 对照表完整性检查；补「未改动章节」说明
- [ ] 可选：`pandoc` 生成 `…_降AIGC.docx` 至 `submission/thesis/`（**不**写回桌面）
- [ ] 更新 `.cursor/goal-thesis-de-aigc.json` 各 Phase 状态

**验收**：用户可用镜像 md vs 降 AIGC md 做 diff；对照表可检索任意改动。

---

## 套话黑名单（优先替换或删除）

| 类型 | 示例 | 建议处理 |
|------|------|----------|
| 递进模板 | 首先…其次…最后… | 改为因果或并列，或删冗余递进 |
| 总结模板 | 综上所述、总而言之 | 删或改为一句具体结论 |
| 空泛背景 | 随着深度学习的发展、在当今社会 | 删或改为赛题/海域一句 |
| 夸大 | 极大地、完美地、显著地（无数据处） | 弱化或附指标 |
| 对称排比 | 不仅…而且…不仅…而且… | 保留一处，其余改写 |
| 被动堆砌 | 被用于…被应用于… | 改主动：本系统采用… |

---

## 关键文件索引

| 用途 | 路径 |
|------|------|
| 定稿 docx（源） | `c:\Users\HxiaoL\Desktop\系统模式论文\基于深度学习的海洋涡旋识别与风浪预警系统设计.docx` |
| 阶段状态 | `.cursor/goal-thesis-de-aigc.json` |
| v6 审阅意见 | `submission/_thesis_v6_review.md` |
| DTW/异常窗改稿提示 | `submission/_thesis_dtw_regional_mean_vs_center_draft.md` |
| 风浪指标口径 | `docs/实验与结果归档/风浪异常_指标口径与台风关联评测.md` |
| 代码事实核对 | `src/anomaly/detect.py`、`src/anomaly/model.py`、`src/eddy/` |
| 镜像 md（产出） | `submission/thesis/基于深度学习的海洋涡旋识别与风浪预警系统设计_原文镜像.md` |
| 工作稿（产出） | `submission/thesis/基于深度学习的海洋涡旋识别与风浪预警系统设计_降AIGC.md` |
| 修订对照（产出） | `submission/thesis/降AIGC修订对照.md` |

---

## 禁止事项（每轮自检）

1. **不得**覆盖或原地修改桌面定稿 docx。  
2. **不得**为降 AIGC 修改实验数字、表号、图号、参考文献。  
3. **不得**把已统一的 `anomaly_index`、区域平均风 DTW、DTW 不参与 3σ 写回旧版「残差 DTW」。  
4. **不得**在系统章重新引入水文/ConvLSTM 作为主模块叙述。  
5. **不得**编造仓库中不存在的 API、模块或实验结果。  
6. 对照表未登记的**大段**改写视为未完成（单章超过 5 处连续改写须有条目）。

---

## 推荐工具命令（Phase 0）

```powershell
# 仓库根目录；需 pandoc 或 pip install python-docx
New-Item -ItemType Directory -Force -Path submission/thesis/assets

pandoc "c:\Users\HxiaoL\Desktop\系统模式论文\基于深度学习的海洋涡旋识别与风浪预警系统设计.docx" `
  -f docx -t gfm --wrap=none --extract-media=submission/thesis/assets `
  -o submission/thesis/基于深度学习的海洋涡旋识别与风浪预警系统设计_原文镜像.md
```

---

## 轮次结束汇报模板

```markdown
## Goal 轮次汇报（thesis-de-aigc）
- Phase：Phase X / 章节 …
- 完成：…
- 变更文件：submission/thesis/…
- 对照表新增条目：N 条
- 技术核对：数值/公式/口径 通过 / 待用户确认 …
- goal-thesis-de-aigc.json：phase=…，checklist 已勾选 …
- 下一步：…
- 阻塞：无 / 需用户查重结果反馈 …
```

---

## 与查重/AIGC 平台的关系

本 Goal **不保证**具体 AIGC 百分比；交付的是「可审计的最小改动工作流」。建议用户在各 Phase 后对 `…_降AIGC.md` 分段送检，将**检出仍高的章节**作为 `/subgoal` 追加精修范围，避免全文反复大改。
