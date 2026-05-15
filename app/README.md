# Streamlit 演示系统

所属项目：**EddyFusion：面向涡旋—水文—风浪的海洋环境智能分析与预警平台**。

## 目标

提供可直接演示的统一入口：**总览**、**实时系统**（视频/RTSP）、**离线系统**（NC 上传摘要）、三模块工作台、**台风知识库**（定调与检索）、指标看板。

## 运行

在仓库根目录执行：

```bash
streamlit run app/main.py
```

或使用一键脚本：

```bash
bash scripts/run_demo.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1
```

## 页面（侧边栏顺序）

- **总览**：三模块状态 + 指标摘要
- **实时系统**：摄像头/RTSP、队列、涡旋限频推理；准实时 NC 接入占位；风浪手动联动
- **离线系统**：多文件 NetCDF 上传、`nc_preprocess_facade` 元数据摘要、三模块同屏占位
- **涡旋识别**：**本页上传视频**、真实模型推理、多通道 NPZ、几何导出
- **水文推理**：L2 默认、示例或上传 NPZ；**NC 探针**（仅摘要）
- **风浪预警**（原「结果」）：涡旋结果联动台风库与报告/LLM
- **台风知识库**：索引、检索、案例 + **系统定调**说明
- **指标看板**：读取 `outputs/` 下 JSON

## 服务与数据路径

- 上传视频缓存：`app/data/media/`
- 上传 NC 缓存：`app/data/nc_uploads/`
- NC 接入：`app/services/nc_ingest_service.py`、`app/services/nc_preprocess_facade.py`

## 说明

- 水文页面优先走真实模型推理（默认 `hydro` L2），并预留 L0 开关；涡旋页与风浪预警页默认走真实推理（依赖 `outputs/eddy/best.pt` 与 `ultralytics`）。
- 统一输入接口为 `InferenceInput`，实时系统与历史会话视频共用推理入口。
- 指标文件缺失不会导致页面崩溃，会显示降级提示。
- 若 `st.video` 报 **Video source error**：多为 mp4 编码非 H.264，见根目录脚本 `scripts/export_eddy_demo_video.py` 的网页兼容转码。
- 风浪预警：无水文残差时可能使用 peak_score 代理；配套 NPZ 含 `demo_wind_*` 时走演示风浪序列（见 `src/anomaly/eddy_typhoon_bridge.py`）。
- 演示包：`submission/figures/eddy_demo/eddy_demo.mp4` + `eddy_demo_physics.npz`（`python scripts/gen_eddy_demo_physics_npz.py`）。
- 演进规划：`相关文件/下一步开发方向_数据入口实时离线与大屏.md`。  
- 第二套前端（地图化水文热力图）：`docs/策略_第二套前端与水文预测热力图.md`（Vite + React/Vue + MapLibre，FastAPI；**不含**本期异常告警地图）。  
- 新前端离线/实时开发手册：`docs/开发文档_新前端离线系统与实时系统.md`。
