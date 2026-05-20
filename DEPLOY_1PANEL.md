# YOLO ONNX + 1Panel 部署流程

本文档面向 `4 核 CPU / 4G 内存 / 30G 磁盘 / Ubuntu 22.04` 腾讯云服务器，部署目标如下：

- 使用 `yolo26n.onnx` 做 CPU 推理
- 支持图片上传、视频上传、浏览器摄像头、视频 URL 识别
- 通过域名访问
- 使用 `1Panel + Docker Compose + 反向代理 + HTTPS`

## 1. 部署前准备

### 1.1 域名与访问方式

- 如果要使用浏览器摄像头，必须走 `HTTPS` 域名访问。
- `http://IP:7860` 可以用于早期联调，但浏览器通常不会给公网 IP 页面开放摄像头权限。
- 建议先准备一个解析到腾讯云公网 IP 的域名，例如 `detect.example.com`。

### 1.2 腾讯云安全组

至少放行以下端口：

- `22`：SSH
- `80`：HTTP
- `443`：HTTPS
- `1Panel 面板端口`：安装后会生成，初次登录面板要用

不建议长期对公网开放 `7860`，后续由 1Panel 反向代理到容器即可。

### 1.3 当前方案支持范围

视频 URL 默认支持：

- `mp4` 直链
- `m3u8`
- `rtsp`
- `http(s)` 视频流

不默认承诺直接支持：

- YouTube 页面链接
- B 站页面链接
- 需要额外解析签名的网页视频地址

## 2. 上传项目文件到服务器

推荐在服务器中使用以下目录：

```bash
mkdir -p /opt/bysj
```

把以下文件上传到 `/opt/bysj`：

- `app_server.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `yolo26n.onnx`

上传完成后，目录结构应类似：

```text
/opt/bysj
├── .dockerignore
├── app_server.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── yolo26n.onnx
```

## 3. 安装 1Panel

在 Ubuntu 22.04 服务器执行：

```bash
bash -c "$(curl -sSL https://resource.fit2cloud.com/1panel/package/v2/quick_start.sh)"
```

安装完成后：

- 记录面板访问地址
- 记录面板端口和安全入口
- 用浏览器登录 1Panel

如果忘了入口信息，可在服务器执行：

```bash
1pctl user-info
```

## 4. 用 Docker Compose 启动识别服务

### 4.1 命令行方式先验证一次

先进入项目目录：

```bash
cd /opt/bysj
```

启动服务：

```bash
docker compose up -d --build
```

如果服务器访问官方 Python 包源较慢，优先使用项目内提供的 `Dockerfile` 新版本，其中已经切换为国内 PyPI 镜像并增大超时时间。

查看日志：

```bash
docker compose logs -f
```

如果看到 `Running on` 或类似 `Uvicorn`/`Gradio` 启动信息，说明容器已正常运行。

### 4.2 浏览器初测

先用下面地址联调：

```text
http://服务器公网IP:7860
```

先测试：

- 图片上传
- 视频上传
- 视频 URL

摄像头功能此时可能仍受浏览器安全策略限制，属于正常现象，等 HTTPS 域名接入后再测。

### 4.3 后续改为 1Panel 托管

你可以继续保留命令行部署，也可以在 1Panel 里用 `Compose` 重新托管这一套服务。

推荐做法：

1. 打开 `1Panel -> 容器 -> Compose`
2. 新建项目
3. 项目路径选择 `/opt/bysj`
4. 内容直接使用当前 `docker-compose.yml`
5. 点击创建并启动

之后的重启、日志查看、重建镜像都能在面板中完成。

## 5. 用 1Panel 绑定域名

### 5.1 创建反向代理网站

在 `1Panel -> 网站 -> 创建网站` 中：

- 创建方式选择 `反向代理`
- 主域名填写你的域名，例如 `detect.example.com`
- 反向代理地址填：

```text
http://127.0.0.1:7860
```

保存后，域名请求就会转发到容器中的 Gradio 服务。

### 5.2 检查域名解析

确保域名 DNS 已经解析到腾讯云服务器公网 IP：

- `A` 记录 -> 服务器 IPv4

如果还没生效，先等待解析完成，再继续申请证书。

## 6. 在 1Panel 中启用 HTTPS

进入 `1Panel -> 网站 -> 证书`：

- 创建或选择 `Acme` 账号
- 申请证书
- 域名填写你的业务域名，例如 `detect.example.com`

推荐两种模式：

### 6.1 HTTP 验证

适用场景：

- 域名已正确解析到当前服务器
- 80 端口已放行

优点：

- 配置简单

### 6.2 DNS 验证

适用场景：

- 80 端口受限制
- 需要申请泛域名证书

优点：

- 更灵活

申请成功后，在网站设置中启用 HTTPS，并开启 HTTP 自动跳转 HTTPS。

## 7. 访问与验证

最终访问地址示例：

```text
https://detect.example.com
```

按以下顺序验证：

1. 图片上传识别
2. 视频上传识别
3. 视频 URL 识别
4. 浏览器摄像头识别

如果摄像头页无法打开，优先检查：

- 是否通过 `HTTPS` 访问
- 当前浏览器是否允许摄像头权限
- 域名证书是否有效

## 8. 日常运维

### 8.1 查看容器状态

```bash
cd /opt/bysj
docker compose ps
```

### 8.2 查看日志

```bash
cd /opt/bysj
docker compose logs -f
```

### 8.3 重建服务

```bash
cd /opt/bysj
docker compose up -d --build
```

### 8.4 停止服务

```bash
cd /opt/bysj
docker compose down
```

## 9. 低配服务器调优建议

当前项目默认已经做了这些限制：

- CPU 推理
- 线程数限制为 `2`
- Gradio 默认单并发
- 默认 `imgsz=512`

如果服务器仍然吃紧，可以继续下调：

- `BYSJ_DEFAULT_IMGSZ=416`
- `BYSJ_DEFAULT_MAX_DET=50`

可以在 `docker-compose.yml` 里直接改环境变量后重启容器。

## 10. 常见问题

### 10.1 容器能启动，但域名打不开

优先检查：

- 域名是否解析到正确公网 IP
- 腾讯云安全组是否放行 `80/443`
- 1Panel 网站是否正确反代到 `127.0.0.1:7860`

### 10.2 摄像头页面打不开或无权限

优先检查：

- 是否是 `HTTPS`
- 浏览器是否阻止了摄像头权限
- 是否通过内嵌 iframe 访问且没有正确放行权限

### 10.3 视频 URL 无法识别

常见原因：

- 输入的是网页播放页而不是视频流地址
- 目标站点有防盗链或鉴权
- 服务器无法访问该 URL

### 10.4 内存不够

优先处理：

- 降低 `imgsz`
- 控制单并发
- 避免多人同时识别视频
- 避免长时间处理高分辨率 RTSP 视频流
