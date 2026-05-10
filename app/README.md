# Streamlit 演示系统

## 目标

提供可直接演示的统一入口：视频上传、涡旋真实推理、水文推理（默认 L2）、台风知识库联动与指标看板读取。

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

## 页面

- 总览：三模块状态 + 指标摘要
- 涡旋识别：真实模型推理、关键帧可视化、检测时间轴与分数统计
- 水文推理：模型/权重路径配置、示例或上传 NPZ 输入、运行单样本推理、图形化结果与指标摘要
- 上传：上传视频并预览，展示元信息
- 结果：运行真实推理，输出结构化时间轴、关键帧，并自动联动台风知识库候选事件
- 台风知识库：查看索引状态、按时间窗与海域查询候选台风、浏览预置联动案例
- 实时输入：摄像头/RTSP 单帧轮询，队列缓冲，推理限频（默认 1fps）
- 指标看板：读取 `outputs/` 下已有 JSON

## 说明

- 水文页面优先走真实模型推理（默认 `hydro` L2），并预留 L0 开关；涡旋页与结果页默认走真实推理（依赖 `outputs/eddy/best.pt` 与 `ultralytics`）。
- 统一输入接口为 `InferenceInput`，上传文件和实时摄像头共用同一推理入口。
- 上传视频缓存目录：`app/data/media/`。
- 指标文件缺失不会导致页面崩溃，会显示降级提示。

