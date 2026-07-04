# Streamlit 演示系统

所属项目：**海洋环境演示（涡旋 / 风浪 / 台风）**，Streamlit 为精简单线入口。

## 目标

侧边栏：**总览**、**涡旋识别**（YOLO，3/7 通道可选）、**风浪预警**、**台风查询**、**指标看板**。  
（实时/离线/水文入口已隐藏；需要水文时设 `EDDYFUSION_SHOW_HYDRO=1` 并自行恢复 `main.py` 中的页面注册。）

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

## 页面（侧边栏）

- **总览**：入口与 `outputs/anomaly` 等指标摘要
- **涡旋识别**：NetCDF → YOLO，**3 通道** 或 **7 通道**（对齐 `build_physics_stacked_hw7` / `outputs/eddy_enh7/`）
- **风浪预警**：风浪 NC 或与涡旋会话联动、`run_detect`、台风检索
- **台风查询**：IBTrACS 构建的 `events.json` 检索
- **指标看板**：读取 `outputs/` JSON

## 服务与数据路径

- 上传视频缓存：`app/data/media/`
- 上传 NC 缓存：`app/data/nc_uploads/`
- NC 接入：`app/services/nc_ingest_service.py`、`app/services/nc_preprocess_facade.py`

## 说明

- 涡旋页 **7 通道** 需提供 `channels=7` 的权重（默认路径 `outputs/eddy_enh7/best.pt`）；**3 通道 Fair-B0** 对应 `outputs/eddy_v6_b0_fair/best.pt`（`config/eddy.yaml`）。
- 指标文件缺失时页面降级提示。
- 若浏览器无法播放 MP4，多为编码非 H.264，请安装 ffmpeg 或通过下载按钮本地播放。
