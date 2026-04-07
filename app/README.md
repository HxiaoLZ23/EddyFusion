# Streamlit 演示系统

## 目标

提供可直接演示的统一入口：视频上传、结果展示（Mock）、指标看板读取。

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
- 上传：上传视频并预览，展示元信息
- 结果：运行 Mock 推理，输出结构化时间轴
- 实时输入：摄像头/RTSP 单帧轮询，队列缓冲，推理限频（默认 1fps）
- 指标看板：读取 `outputs/` 下已有 JSON

## 说明

- 当前默认 `mock` 推理，真实推理适配层已在 `app/services/inference_service.py` 预留。
- 统一输入接口为 `InferenceInput`，上传文件和实时摄像头共用同一推理入口。
- 上传视频缓存目录：`app/data/media/`。
- 指标文件缺失不会导致页面崩溃，会显示降级提示。

