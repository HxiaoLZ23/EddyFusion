# 论文系统功能测试表（§6.3 T1～T8，同表 6-7）

> 由 `python scripts/run_system_tests.py --skip-ui` 自动生成；用例实现见 `tests/test_paper_system_api.py`。
> 测试数据为仓库内合成小 NC（`demo_eddy_nc` / `demo_windwave_nc` fixture），验证 FastAPI 演示链路。

**表 6-7  系统功能测试结果**

| 编号 | 测试项 | 接口/方法 | 预期结果 | 测试结果 | 耗时(s) |
| --- | --- | --- | --- | --- | --- |
| T1 | NC 元数据探测 | `GET /api/preprocess/meta` | 返回 `time_len`、`variables`、`variable_map`（含 `eddy_ready`） | 通过 | 2.260 |
| T2 | 时空裁剪（ROI + 时间索引） | `POST /api/preprocess/subset` | 子集 NC 写入 `app/data/nc_uploads/subsets/` | 通过 | 0.613 |
| T3 | 涡旋单帧预览 | `POST /api/eddy/preview-frame` | PNG `data URL` + `stats_rows`（YOLO 或 ADT 降级） | 通过 | 3.824 |
| T4 | 涡旋双路 MP4（异步分阶段） | `POST /api/jobs`（`eddy_dual_mp4`） | job 至 `done`；底图路与标注路 MP4 可访问 | 通过 | 4.530 |
| T5 | 风浪预测（同步） | `POST /api/windwave/forecast` | `series`、`anomaly_segments`、`typhoon_candidates`、异常等级 | 通过 | 1.060 |
| T6 | 结构化报告归档 | `POST /api/report/structured` → `save` → `history` | 可列表、按 id 读取 Markdown 正文 | 通过 | 1.085 |
| T7 | 风浪预测（异步 job） | `POST /api/jobs`（`windwave_forecast`） | 轮询至 `done`，`result.series` 非空 | 通过 | 1.623 |
| T8 | 准实时连接器状态 | `GET /api/realtime/status` | `connected`、`poll_dir`、`source` 等字段完整 | 通过 | 0.016 |

**汇总**：8 通过 / 0 跳过 / 8 项（T4 无本地 3ch 权重时跳过，不影响其余项）。

复现命令：

```powershell
python -m pytest tests/test_paper_system_api.py -v
# 或
python scripts/run_system_tests.py --skip-ui
```

说明：本表为 **API 自动化**口径；前端页面（监测总览上传、报告管理 LLM 解读等）见 §5 界面截图与答辩演示脚本。
