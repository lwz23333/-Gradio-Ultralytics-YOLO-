from __future__ import annotations

import os
import csv
import io
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO


# Conservative defaults for a 4C/4G CPU-only server.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo26n.onnx"
FIXED_ONNX_IMGSZ = int(os.environ.get("BYSJ_ONNX_IMGSZ", "416"))
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = OUTPUT_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
_ACCELERATION_DISABLED = False


def _get_model_path() -> Path:
    raw = (
        os.environ.get("BYSJ_MODEL")
        or os.environ.get("BYSJ_WEIGHTS")
        or ""
    ).strip().strip('"').strip("'")
    return Path(raw) if raw else DEFAULT_MODEL


@lru_cache(maxsize=1)
def _has_usable_gpu() -> bool:
    try:
        import onnxruntime as ort

        has_cuda_provider = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        has_cuda_provider = False

    if not has_cuda_provider:
        return False

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        completed = subprocess.run(
            [nvidia_smi, "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _get_device() -> str:
    raw = (os.environ.get("BYSJ_DEVICE") or "").strip().lower()
    if raw:
        return "cpu" if raw == "cpu" else raw
    if _ACCELERATION_DISABLED:
        return "cpu"
    return "0" if _has_usable_gpu() else "cpu"


def _device_label() -> str:
    raw = (os.environ.get("BYSJ_DEVICE") or "").strip().lower()
    if raw:
        return "cpu" if raw == "cpu" else f"{raw}（手动指定）"
    if _ACCELERATION_DISABLED:
        return "cpu（GPU 推理失败后自动回退）"
    if _has_usable_gpu():
        return "GPU+CPU（自动，CUDAExecutionProvider + CPUExecutionProvider）"
    return "cpu"


@lru_cache(maxsize=1)
def _load_model() -> YOLO:
    model_path = _get_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


def _placeholder(message: str) -> tuple[np.ndarray, str, str, dict[str, int]]:
    return np.zeros((8, 8, 3), dtype=np.uint8), message, "总目标数: 0", {}


def _auto_predict_params(source: object, conf: float, iou: float, max_det: int) -> tuple[float, float, int, int]:
    used_conf = float(conf)
    used_iou = float(iou)
    used_max_det = int(max_det)

    if isinstance(source, np.ndarray) and source.ndim >= 2:
        height, width = source.shape[:2]
        long_edge = max(height, width)
        if long_edge >= 2500:
            used_conf = min(used_conf, 0.10)
            used_max_det = max(used_max_det, 200)
        elif long_edge >= 1600:
            used_conf = min(used_conf, 0.15)
            used_max_det = max(used_max_det, 150)

    return used_conf, used_iou, FIXED_ONNX_IMGSZ, used_max_det


def _run_predict(source: object, conf: float, iou: float, imgsz: int, max_det: int):
    global _ACCELERATION_DISABLED

    model = _load_model()
    used_conf, used_iou, used_imgsz, used_max_det = _auto_predict_params(source, conf, iou, max_det)
    device = _get_device()
    predict_kwargs = dict(
        source=source,
        conf=used_conf,
        iou=used_iou,
        imgsz=used_imgsz,
        max_det=used_max_det,
        verbose=False,
    )
    try:
        return model.predict(device=device, **predict_kwargs)
    except Exception:
        if device == "cpu":
            raise
        _ACCELERATION_DISABLED = True
        return model.predict(device="cpu", **predict_kwargs)


def _extract_counts(result) -> tuple[int, dict[str, int], list[str]]:
    if result.boxes is None or len(result.boxes) == 0:
        return 0, {}, []

    names = result.names or {}
    cls = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    confs = result.boxes.conf.detach().cpu().numpy().tolist()

    counts = Counter(str(names.get(c, c)) for c in cls)
    detail_lines = [
        f"- {names.get(c, str(c))}: conf={score:.3f}"
        for c, score in zip(cls[:50], confs[:50])
    ]
    if len(cls) > 50:
        detail_lines.append(f"... 仅展示前 50 个，共 {len(cls)} 个")

    return len(cls), dict(counts.most_common()), detail_lines


def _extract_detections(result) -> list[dict[str, object]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    names = result.names or {}
    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(float).tolist()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    confs = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()

    return [
        {
            "box": box,
            "class_name": str(names.get(class_id, class_id)),
            "conf": conf,
        }
        for box, class_id, conf in zip(xyxy, cls, confs)
    ]


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _center_distance_ratio(a: list[float], b: list[float]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return distance / scale


def _area_ratio(a: list[float], b: list[float]) -> float:
    area_a = max(1.0, max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]))
    area_b = max(1.0, max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    small = min(area_a, area_b)
    large = max(area_a, area_b)
    return small / large


def _match_score(track_box: list[float], det_box: list[float]) -> float:
    iou_score = _box_iou(track_box, det_box)
    center_score = max(0.0, 1.0 - _center_distance_ratio(track_box, det_box))
    size_score = _area_ratio(track_box, det_box)
    return max(iou_score, center_score * size_score)


def _add_track_vote(track: dict[str, object], class_name: str, confidence: float) -> None:
    votes = track.setdefault("class_votes", {})
    if isinstance(votes, dict):
        votes[class_name] = float(votes.get(class_name, 0.0)) + max(float(confidence), 0.01)
    hit_counts = track.setdefault("class_hits", {})
    if isinstance(hit_counts, dict):
        hit_counts[class_name] = int(hit_counts.get(class_name, 0)) + 1


def _dominant_track_class(track: dict[str, object]) -> str:
    votes = track.get("class_votes")
    if isinstance(votes, dict) and votes:
        return str(max(votes.items(), key=lambda item: float(item[1]))[0])
    return str(track.get("class_name", "unknown"))


def _update_video_tracks(
    active_tracks: list[dict[str, object]],
    all_tracks: list[dict[str, object]],
    detections: list[dict[str, object]],
    frame_index: int,
    next_track_id: int,
) -> int:
    match_threshold = float(os.environ.get("BYSJ_VIDEO_TRACK_MATCH", "0.30"))
    cross_class_threshold = float(os.environ.get("BYSJ_VIDEO_TRACK_CROSS_CLASS_MATCH", "0.45"))
    max_age = int(os.environ.get("BYSJ_VIDEO_TRACK_MAX_AGE", "20"))
    matched_track_indexes: set[int] = set()

    for detection in detections:
        best_index = -1
        best_score = 0.0
        det_box = detection["box"]
        for index, track in enumerate(active_tracks):
            if index in matched_track_indexes:
                continue
            score = _match_score(track["box"], det_box)
            if _dominant_track_class(track) == detection["class_name"]:
                score += 0.08
            if score > best_score:
                best_score = score
                best_index = index

        if best_index >= 0:
            best_track = active_tracks[best_index]
            threshold = (
                match_threshold
                if _dominant_track_class(best_track) == detection["class_name"]
                else cross_class_threshold
            )
        else:
            threshold = match_threshold

        if best_index >= 0 and best_score >= threshold:
            track = active_tracks[best_index]
            track["box"] = det_box
            track["last_seen"] = frame_index
            track["hits"] = int(track["hits"]) + 1
            track["max_conf"] = max(float(track["max_conf"]), float(detection["conf"]))
            _add_track_vote(track, str(detection["class_name"]), float(detection["conf"]))
            track["class_name"] = _dominant_track_class(track)
            matched_track_indexes.add(best_index)
        else:
            track = {
                "id": next_track_id,
                "class_name": detection["class_name"],
                "box": det_box,
                "first_seen": frame_index,
                "last_seen": frame_index,
                "hits": 1,
                "max_conf": float(detection["conf"]),
            }
            _add_track_vote(track, str(detection["class_name"]), float(detection["conf"]))
            next_track_id += 1
            active_tracks.append(track)
            all_tracks.append(track)

    active_tracks[:] = [
        track
        for track in active_tracks
        if frame_index - int(track["last_seen"]) <= max_age
    ]
    return next_track_id


def _summarize_unique_tracks(all_tracks: list[dict[str, object]]) -> tuple[int, dict[str, int]]:
    min_hits = int(os.environ.get("BYSJ_VIDEO_TRACK_MIN_HITS", "4"))
    min_conf = float(os.environ.get("BYSJ_VIDEO_TRACK_MIN_CONF", "0.25"))
    stable_tracks = [
        track
        for track in all_tracks
        if int(track["hits"]) >= min_hits and float(track["max_conf"]) >= min_conf
    ]
    raw_counts = Counter(_dominant_track_class(track) for track in stable_tracks)
    if not stable_tracks:
        return 0, {}

    dominant_count = max(raw_counts.values())
    rare_min_hits = int(os.environ.get("BYSJ_VIDEO_RARE_CLASS_MIN_HITS", "10"))
    rare_min_conf = float(os.environ.get("BYSJ_VIDEO_RARE_CLASS_MIN_CONF", "0.65"))
    filtered_tracks = []
    for track in stable_tracks:
        class_name = _dominant_track_class(track)
        is_main_class = raw_counts[class_name] == dominant_count
        has_strong_evidence = (
            int(track["hits"]) >= rare_min_hits
            or float(track["max_conf"]) >= rare_min_conf
            or dominant_count <= 1
        )
        if is_main_class or has_strong_evidence:
            filtered_tracks.append(track)

    counts = Counter(_dominant_track_class(track) for track in filtered_tracks)
    return len(filtered_tracks), dict(counts.most_common())


def _format_summary(result, prefix: str) -> tuple[str, str, dict[str, int]]:
    total, counts, detail_lines = _extract_counts(result)
    lines = [
        prefix,
        f"model={_get_model_path()}",
        f"device={_device_label()}",
        f"总目标数={total}",
    ]
    if counts:
        lines.append("按类别统计:")
        lines.extend(f"- {name}: {count}" for name, count in counts.items())
        lines.extend(detail_lines)
    else:
        lines.append("未检测到目标。")
    return "\n".join(lines), f"总目标数: {total}", counts


def _normalize_counts(counts: object) -> dict[str, int]:
    if not counts:
        return {}
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items()}
    if isinstance(counts, str):
        try:
            parsed = json.loads(counts)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): int(value) for key, value in parsed.items()}
    return {}


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _cleanup_generated_files(keep_paths: set[Path] | None = None) -> str:
    keep = {path.resolve() for path in (keep_paths or set()) if path}
    targets: list[Path] = []
    targets.extend(OUTPUT_DIR.glob("bysj_video_*.mp4"))
    targets.extend(OUTPUT_DIR.glob("bysj_video_raw_*.mp4"))
    targets.extend(EXPORT_DIR.glob("bysj_export_*.zip"))
    top_pycache = ROOT / "__pycache__"
    if top_pycache.exists():
        targets.append(top_pycache)

    deleted = 0
    freed = 0
    errors: list[str] = []
    for target in targets:
        try:
            if target.resolve() in keep:
                continue
            size = _path_size(target)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            deleted += 1
            freed += size
        except Exception as exc:
            errors.append(f"{target.name}: {exc}")

    message = f"已清理 {deleted} 个缓存/输出文件，释放约 {freed / 1024 / 1024:.2f} MB。"
    if errors:
        message += "\n未能清理: " + "; ".join(errors[:5])
    return message


def clear_output_cache() -> str:
    return _cleanup_generated_files()


@lru_cache(maxsize=1)
def _get_ffmpeg_executable() -> str | None:
    explicit = (os.environ.get("BYSJ_FFMPEG") or "").strip().strip('"').strip("'")
    if explicit and Path(explicit).is_file():
        return explicit

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        return bundled if bundled and Path(bundled).is_file() else None
    except Exception:
        return None


def _make_browser_playable_mp4(raw_path: str) -> tuple[str, str]:
    raw = Path(raw_path)
    ffmpeg = _get_ffmpeg_executable()
    if not ffmpeg:
        return raw_path, "未找到 ffmpeg，检测结果已生成但浏览器可能无法直接预览；可使用导出功能下载。"

    playable = raw.with_name(raw.name.replace("bysj_video_raw_", "bysj_video_"))
    if playable == raw:
        playable = raw.with_name(f"{raw.stem}_playable.mp4")

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(playable),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(os.environ.get("BYSJ_FFMPEG_TIMEOUT", "1800")),
            check=False,
        )
    except Exception as exc:
        return raw_path, f"视频转码失败，已保留原始检测视频；原因: {exc}"

    if completed.returncode == 0 and playable.is_file() and playable.stat().st_size > 0:
        raw.unlink(missing_ok=True)
        return str(playable), "检测视频已转为浏览器兼容的 H.264 MP4。"

    detail = (completed.stderr or "").strip().splitlines()
    last_line = detail[-1] if detail else "未知错误"
    return raw_path, f"视频转码失败，已保留原始检测视频；原因: {last_line}"


def _coerce_file_path(value: object) -> Path | None:
    if not value:
        return None
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    if isinstance(value, dict):
        for key in ("video", "path", "name"):
            candidate = value.get(key)
            if candidate:
                return Path(candidate)
    if isinstance(value, (list, tuple)) and value:
        return _coerce_file_path(value[0])
    return None


def _export_result(
    kind: str,
    summary: str | None,
    total_text: str | None,
    counts: object,
    video_path: object = None,
    cleanup_after: bool = False,
) -> tuple[str | None, str]:
    normalized_counts = _normalize_counts(counts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORT_DIR / f"bysj_export_{kind}_{timestamp}.zip"

    metadata = {
        "kind": kind,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(_get_model_path()),
        "device": _device_label(),
        "total": total_text or "",
        "counts": normalized_counts,
    }

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["class", "count"])
    for name, count in normalized_counts.items():
        writer.writerow([name, count])

    summary_text = summary or "暂无识别摘要。"
    if total_text:
        summary_text = f"{summary_text}\n\n{total_text}"

    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.txt", summary_text)
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("counts.csv", csv_buffer.getvalue())

        candidate = _coerce_file_path(video_path)
        if candidate and candidate.is_file():
            archive.write(candidate, arcname=candidate.name)

    cleanup_status = ""
    if cleanup_after:
        cleanup_status = "\n" + _cleanup_generated_files(keep_paths={export_path})

    return str(export_path), f"导出完成: {export_path}{cleanup_status}"


def export_image_result(summary: str, total_text: str, counts: object, cleanup_after: bool):
    return _export_result("image", summary, total_text, counts, cleanup_after=cleanup_after)


def export_video_result(video_path: str, summary: str, total_text: str, counts: object, cleanup_after: bool):
    return _export_result("video", summary, total_text, counts, video_path=video_path, cleanup_after=cleanup_after)


def export_webcam_result(summary: str, total_text: str, counts: object, cleanup_after: bool):
    return _export_result("webcam", summary, total_text, counts, cleanup_after=cleanup_after)


def export_url_result(summary: str, total_text: str, counts: object, cleanup_after: bool):
    return _export_result("url", summary, total_text, counts, cleanup_after=cleanup_after)


def predict_image(
    image: np.ndarray | None,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> tuple[np.ndarray, str, str, dict[str, int]]:
    if image is None:
        return _placeholder("请先上传图片。")

    result = _run_predict(image, conf, iou, imgsz, max_det)[0]
    plotted = result.plot()[:, :, ::-1]
    summary, total_text, counts = _format_summary(result, "图片识别完成")
    return plotted, summary, total_text, counts


def predict_webcam(
    image: np.ndarray | None,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> tuple[np.ndarray | None, str, str, dict[str, int]]:
    if image is None:
        return None, "等待摄像头画面。", "总目标数: 0", {}

    start = time.perf_counter()
    result = _run_predict(image, conf, iou, imgsz, max_det)[0]
    plotted = result.plot()
    fps = 1.0 / max(time.perf_counter() - start, 1e-6)
    cv2.putText(
        plotted,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    summary, total_text, counts = _format_summary(result, f"摄像头实时识别中，当前 FPS={fps:.1f}")
    return plotted[:, :, ::-1], summary, total_text, counts


def predict_video_file(
    video_path: str | None,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> tuple[str | None, str, str, dict[str, int]]:
    if not video_path:
        return None, "请先上传视频文件。", "总目标数: 0", {}

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None, f"无法打开视频文件: {video_path}", "总目标数: 0", {}

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        return None, "无法读取视频分辨率。", "总目标数: 0", {}

    fd, output_path = tempfile.mkstemp(prefix="bysj_video_raw_", suffix=".mp4", dir=OUTPUT_DIR)
    os.close(fd)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    hit_frames = 0
    frame_box_total = 0
    active_tracks: list[dict[str, object]] = []
    all_tracks: list[dict[str, object]] = []
    next_track_id = 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = _run_predict(frame, conf, iou, imgsz, max_det)[0]
            writer.write(result.plot())

            total, _, _ = _extract_counts(result)
            detections = _extract_detections(result)
            frame_count += 1
            frame_box_total += total
            if total > 0:
                hit_frames += 1
            next_track_id = _update_video_tracks(
                active_tracks,
                all_tracks,
                detections,
                frame_count,
                next_track_id,
            )
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        return None, "视频中没有可处理的帧。", "总目标数: 0", {}

    unique_total, unique_counts = _summarize_unique_tracks(all_tracks)
    min_hits = int(os.environ.get("BYSJ_VIDEO_TRACK_MIN_HITS", "4"))
    match_threshold = float(os.environ.get("BYSJ_VIDEO_TRACK_MATCH", "0.30"))
    rare_min_hits = int(os.environ.get("BYSJ_VIDEO_RARE_CLASS_MIN_HITS", "10"))
    rare_min_conf = float(os.environ.get("BYSJ_VIDEO_RARE_CLASS_MIN_CONF", "0.65"))
    lines = [
        "视频识别完成",
        f"model={_get_model_path()}",
        f"device={_device_label()}",
        f"总帧数={frame_count}",
        f"检测到目标的帧数={hit_frames}",
        f"逐帧检测框累计数={frame_box_total}",
        f"去重目标数={unique_total}",
        f"去重参数: match={match_threshold:.2f}, min_hits={min_hits}",
        f"罕见类别过滤: rare_min_hits={rare_min_hits}, rare_min_conf={rare_min_conf:.2f}",
    ]
    if unique_counts:
        lines.append("去重类别统计:")
        lines.extend(f"- {name}: {count}" for name, count in unique_counts.items())
    else:
        lines.append("整个视频未检测到目标。")

    playable_path, transcode_status = _make_browser_playable_mp4(output_path)
    lines.append(transcode_status)

    return playable_path, "\n".join(lines), f"去重目标数: {unique_total}", unique_counts


def predict_video_url(
    url: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
):
    global _ACCELERATION_DISABLED

    clean_url = (url or "").strip()
    if not clean_url:
        yield None, "请输入视频 URL。", "总目标数: 0", {}
        return

    model = _load_model()
    device = _get_device()
    predict_kwargs = dict(
        source=clean_url,
        stream=True,
        conf=conf,
        iou=iou,
        imgsz=FIXED_ONNX_IMGSZ,
        max_det=max_det,
        verbose=False,
    )

    try:
        results = model.predict(device=device, **predict_kwargs)
        for index, result in enumerate(results, start=1):
            summary, total_text, counts = _format_summary(result, f"视频流识别中，第 {index} 帧")
            yield result.plot()[:, :, ::-1], summary, total_text, counts
    except Exception:
        if device == "cpu":
            raise
        _ACCELERATION_DISABLED = True
        results = model.predict(device="cpu", **predict_kwargs)
        for index, result in enumerate(results, start=1):
            summary, total_text, counts = _format_summary(result, f"视频流识别中，第 {index} 帧")
            yield result.plot()[:, :, ::-1], summary, total_text, counts


def build_ui() -> gr.Blocks:
    default_imgsz = FIXED_ONNX_IMGSZ
    default_conf = float(os.environ.get("BYSJ_DEFAULT_CONF", "0.25"))
    default_iou = float(os.environ.get("BYSJ_DEFAULT_IOU", "0.70"))
    default_max_det = int(os.environ.get("BYSJ_DEFAULT_MAX_DET", "100"))

    with gr.Blocks(title="YOLO ONNX 识别平台") as demo:
        gr.Markdown(
            "## YOLO ONNX 识别平台\n"
            f"- 模型文件: `{_get_model_path()}`\n"
            f"- 推理设备: `{_device_label()}`\n"
            "- 检测到可用 NVIDIA GPU 时自动使用 GPU+CPU 推理，失败时回退 CPU。\n"
            "- 浏览器摄像头功能需要 HTTPS 域名访问。"
        )

        with gr.Row():
            conf = gr.Slider(0.05, 0.90, value=default_conf, step=0.01, label="conf")
            iou = gr.Slider(0.10, 0.95, value=default_iou, step=0.01, label="iou")
            imgsz = gr.Number(value=default_imgsz, label="imgsz（ONNX固定）", precision=0, interactive=False)
            max_det = gr.Slider(1, 200, value=default_max_det, step=1, label="max_det")

        with gr.Row():
            cleanup_after_export = gr.Checkbox(
                value=False,
                label="导出后自动清理旧输出缓存",
            )
            clear_cache_btn = gr.Button("手动清理缓存", variant="secondary")
        clear_cache_status = gr.Textbox(label="缓存清理状态", lines=2)
        clear_cache_btn.click(fn=clear_output_cache, outputs=clear_cache_status)

        with gr.Tabs():
            with gr.Tab("图片上传"):
                with gr.Row():
                    image_input = gr.Image(
                        type="numpy",
                        label="输入图片",
                        sources=["upload", "clipboard"],
                    )
                    image_output = gr.Image(type="numpy", label="检测结果")
                image_total = gr.Textbox(label="总数", lines=1)
                image_counts = gr.JSON(label="类别统计")
                image_summary = gr.Textbox(label="结果摘要", lines=12)
                image_btn = gr.Button("开始图片识别", variant="primary")
                with gr.Row():
                    image_export_btn = gr.Button("导出图片识别结果", variant="secondary")
                    image_export_file = gr.File(label="导出文件")
                image_export_status = gr.Textbox(label="导出状态", lines=2)
                image_btn.click(
                    fn=predict_image,
                    inputs=[image_input, conf, iou, imgsz, max_det],
                    outputs=[image_output, image_summary, image_total, image_counts],
                )
                image_export_btn.click(
                    fn=export_image_result,
                    inputs=[image_summary, image_total, image_counts, cleanup_after_export],
                    outputs=[image_export_file, image_export_status],
                )

            with gr.Tab("视频上传"):
                video_input = gr.Video(sources=["upload"], label="输入视频")
                video_output = gr.Video(label="检测结果视频")
                video_total = gr.Textbox(label="总数", lines=1)
                video_counts = gr.JSON(label="类别统计")
                video_summary = gr.Textbox(label="结果摘要", lines=12)
                video_btn = gr.Button("开始视频识别", variant="primary")
                with gr.Row():
                    video_export_btn = gr.Button("导出视频识别结果", variant="secondary")
                    video_export_file = gr.File(label="导出文件")
                video_export_status = gr.Textbox(label="导出状态", lines=2)
                video_btn.click(
                    fn=predict_video_file,
                    inputs=[video_input, conf, iou, imgsz, max_det],
                    outputs=[video_output, video_summary, video_total, video_counts],
                )
                video_export_btn.click(
                    fn=export_video_result,
                    inputs=[video_output, video_summary, video_total, video_counts, cleanup_after_export],
                    outputs=[video_export_file, video_export_status],
                )

            with gr.Tab("摄像头实时识别"):
                with gr.Row():
                    webcam_input = gr.Image(
                        type="numpy",
                        sources=["webcam"],
                        streaming=True,
                        label="摄像头画面",
                    )
                    webcam_output = gr.Image(type="numpy", label="检测结果")
                webcam_total = gr.Textbox(label="总数", lines=1)
                webcam_counts = gr.JSON(label="类别统计")
                webcam_summary = gr.Textbox(label="结果摘要", lines=10)
                with gr.Row():
                    webcam_export_btn = gr.Button("导出摄像头识别结果", variant="secondary")
                    webcam_export_file = gr.File(label="导出文件")
                webcam_export_status = gr.Textbox(label="导出状态", lines=2)
                webcam_input.stream(
                    fn=predict_webcam,
                    inputs=[webcam_input, conf, iou, imgsz, max_det],
                    outputs=[webcam_output, webcam_summary, webcam_total, webcam_counts],
                )
                webcam_export_btn.click(
                    fn=export_webcam_result,
                    inputs=[webcam_summary, webcam_total, webcam_counts, cleanup_after_export],
                    outputs=[webcam_export_file, webcam_export_status],
                )

            with gr.Tab("视频 URL"):
                url_input = gr.Textbox(
                    label="视频 URL",
                    placeholder="支持 mp4、m3u8、rtsp、http(s) 视频流直链",
                    lines=1,
                )
                with gr.Row():
                    url_start = gr.Button("开始识别", variant="primary")
                    url_stop = gr.Button("停止识别", variant="secondary")
                url_output = gr.Image(type="numpy", label="实时检测结果")
                url_total = gr.Textbox(label="总数", lines=1)
                url_counts = gr.JSON(label="类别统计")
                url_summary = gr.Textbox(label="结果摘要", lines=10)
                with gr.Row():
                    url_export_btn = gr.Button("导出 URL 识别结果", variant="secondary")
                    url_export_file = gr.File(label="导出文件")
                url_export_status = gr.Textbox(label="导出状态", lines=2)
                url_event = url_start.click(
                    fn=predict_video_url,
                    inputs=[url_input, conf, iou, imgsz, max_det],
                    outputs=[url_output, url_summary, url_total, url_counts],
                )
                url_stop.click(fn=None, cancels=[url_event])
                url_export_btn.click(
                    fn=export_url_result,
                    inputs=[url_summary, url_total, url_counts, cleanup_after_export],
                    outputs=[url_export_file, url_export_status],
                )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue(max_size=4, default_concurrency_limit=1)
    ui.launch(
        server_name=os.environ.get("BYSJ_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("BYSJ_PORT", "7860")),
        show_error=True,
    )
