# YOLO26n ONNX 识别平台

这是一个基于 Gradio、Ultralytics 和 ONNX Runtime 的 YOLO26n 目标识别项目，支持图片上传、视频上传、浏览器摄像头实时识别、视频 URL 识别、结果导出和缓存清理。

## 核心功能

- 图片识别：上传图片后返回检测框、总数、类别统计和摘要。
- 视频识别：处理本地视频，输出浏览器可播放的 H.264 MP4，并使用跨帧轨迹去重减少重复计数。
- 摄像头识别：支持浏览器摄像头单帧实时识别。公网摄像头权限通常需要 HTTPS。
- 视频 URL：支持可被 OpenCV 读取的 HTTP/HTTPS 视频直链、视频流或 RTSP。
- 推理设备：本地检测到可用 NVIDIA GPU 时自动优先使用 GPU，并保留 CPU 回退；服务器可用 `BYSJ_DEVICE=cpu` 固定 CPU。
- 导出与清理：可导出 ZIP 结果包，包含摘要、类别统计 CSV、元数据和检测后视频；支持手动或导出后自动清理旧输出缓存。

## 文件说明

```text
app_server.py        Gradio 页面和推理主程序
requirements.txt     Python 依赖
Dockerfile           服务器 Docker 镜像构建文件
docker-compose.yml   Docker Compose 部署配置
.dockerignore        Docker 构建忽略规则
.gitignore           Git 忽略规则
DEPLOY_1PANEL.md     1Panel + Docker Compose 部署说明
start_project.ps1    Windows 一键启动脚本
一键启动项目.bat      Windows 双击启动入口
```

模型文件 `yolo26n.onnx` 体积较大，默认不提交到 Git。运行前请把模型放到项目根目录，或通过环境变量 `BYSJ_MODEL` 指定模型路径。

## Windows 本地运行

```powershell
cd D:\bysj
D:\python\python.exe -m pip install -r requirements.txt
$env:BYSJ_HOST = "127.0.0.1"
$env:BYSJ_PORT = "7860"
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = "localhost,127.0.0.1"
D:\python\python.exe app_server.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

也可以双击 `一键启动项目.bat`。

## Docker 部署

把 `yolo26n.onnx` 放到项目根目录后执行：

```bash
docker compose up -d --build
```

默认端口：

```text
http://服务器IP:7860
```

服务器上若只更新 `app_server.py`，通常执行：

```bash
cd /opt/bysj
sudo docker compose restart
```

如果镜像内代码未更新，再执行：

```bash
sudo docker compose up -d --build
```

## 常用环境变量

```text
BYSJ_MODEL=/path/to/yolo26n.onnx
BYSJ_HOST=127.0.0.1
BYSJ_PORT=7860
BYSJ_DEVICE=cpu
BYSJ_ONNX_IMGSZ=416
BYSJ_DEFAULT_CONF=0.25
BYSJ_DEFAULT_IOU=0.70
BYSJ_DEFAULT_MAX_DET=100
```
