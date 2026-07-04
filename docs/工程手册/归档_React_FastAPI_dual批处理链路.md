# 归档：React/FastAPI 涡旋 Dual 批处理链路

> 目的：固定 `web/` + `web_api/` 在涡旋视频生成上的**高性能实现事实**，并与 Streamlit 端对齐，避免后续回退到逐帧串行逻辑。  
> 结论：Dual 链路 = **批量抽帧（可多进程）+ 批量 YOLO + 分阶段编码**，比逐帧 `infer_netcdf_frame` 循环更适合长时序 NC。

---

## 1. 入口与文件

| 层 | 文件/接口 | 作用 |
|---|---|---|
| Web API 路由 | `web_api/routers/eddy_dual.py` | 暴露 `/api/eddy/dual-mp4` 与 `/api/eddy/dual-mp4/annotate` |
| 核心服务 | `app/services/eddy_demo_service.py` | `infer_netcdf_dual_mp4()`、`complete_dual_mp4_from_job()` |
| 批处理抽帧 | `src/eddy/nc_dual_batch.py` | `extract_triple_slices_batch()` + `build_dual_frames_parallel()` |
| 预览播放 | `GET /api/eddy/preview?file=...` | 播放 `eddy_nc_base_*.mp4` / `eddy_nc_ann_*.mp4` |

---

## 2. 双阶段流程（实际执行）

1. **规划帧索引**：`_plan_dual_indices()`  
   - 自动限制最大推理帧数（受 `max_frames` 和 `EDDY_DUAL_MAX_INFER_FRAMES` 双重约束）
2. **批量抽帧与预构建**：`_cache_dual_frames()`  
   - 一次性打开/读取 NC 时间片  
   - 生成并缓存 `base_*.npy`、`yolo_*.npy`、可选 `plot_*.npy`
3. **先编码底图视频**：`eddy_nc_base_*.mp4`
4. **批量 YOLO 推理并编码带框视频**：`complete_dual_mp4_from_job()`  
   - `model.predict(chunk)` 按 batch 处理，而不是每帧单调
5. **返回产物**：`base_mp4` + `annotated_mp4` + `detection_timeline`

---

## 3. API 口径

### `POST /api/eddy/dual-mp4`

- `deliver=full`：完整双路（默认）
- `deliver=base`：先返回底图，后续再 `annotate`
- `deliver=annotate`：不在此接口使用（会返回 400，需走专门接口）

### `POST /api/eddy/dual-mp4/annotate`

- 输入已有 `job_id`
- 对缓存帧执行第二阶段 YOLO 与标注视频编码

---

## 4. 性能开关（环境变量）

| 变量 | 默认 | 含义 |
|---|---:|---|
| `EDDY_DUAL_EXTRACT_WORKERS` | `0` | 抽帧并行度：`0` 自动，`1` 关闭多进程，`>1` 固定进程数 |
| `EDDY_DUAL_YOLO_BATCH` | `4` | 批量 YOLO 的 chunk 大小 |
| `EDDY_DUAL_MAX_INFER_FRAMES` | `120` | dual 推理硬上限（防止超长任务） |
| `EDDY_YOLO_DEVICE` | 空 | YOLO 设备覆盖（如 `0`、`cpu`） |

---

## 5. 与旧串行方案对比

| 方案 | 入口 | 特征 | 适用 |
|---|---|---|---|
| 旧串行 | `infer_netcdf_detection_video()` | `for ti -> infer_netcdf_frame` 逐帧推理 | 小样本、临时调试 |
| Dual 批处理 | `infer_netcdf_dual_mp4()` | 批抽帧 + 批 YOLO + 分阶段编码 | 长时序 NC、演示主路径 |

---

## 6. Streamlit 对齐说明（2026-05-28）

- `app/pages/eddy.py` 已切到 `infer_netcdf_dual_mp4(deliver="full")` 生成视频；
- 会话联动（`eddy_last_result`）仍单独推 1 帧 `infer_netcdf_frame()`，用于几何与风浪页读取；
- 因此现在 Streamlit 与 React/FastAPI 在**视频生成链路**上已同口径。

---

## 7. 快速排障

- **慢**：先检查 `EDDY_DUAL_EXTRACT_WORKERS`、`EDDY_DUAL_YOLO_BATCH`、`EDDY_YOLO_DEVICE` 是否生效。  
- **浏览器播不了**：通常 ffmpeg 不可用退回 `mp4v`，安装 ffmpeg 并加入 PATH。  
- **带框视频缺失**：检查 `job_id` 对应的 `app/data/eddy_preview/jobs/<job_id>/manifest.json` 与 `yolo_*.npy` 是否完整。  
- **通道不匹配**：3ch/7ch/8ch 权重要和首层输入通道一致（尤其 7ch 使用 `outputs/eddy_enh7/*`）。

