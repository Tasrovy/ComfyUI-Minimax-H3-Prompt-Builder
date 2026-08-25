import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid

from aiohttp import web
import folder_paths
from server import PromptServer


FORMAT = "minimax-h3-reference-video-asset"
VERSION = 1
FPS = 24
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,96}$")


def _asset_directories():
    metadata = os.path.join(folder_paths.get_user_directory(), "minimax_h3", "reference_videos")
    media = os.path.join(folder_paths.get_input_directory(), "minimax_h3", "reference_videos")
    os.makedirs(metadata, exist_ok=True)
    os.makedirs(media, exist_ok=True)
    return metadata, media


def _asset_path(asset_id):
    if not ID_PATTERN.fullmatch(asset_id):
        raise ValueError("参考视频资产 ID 格式错误")
    return os.path.join(_asset_directories()[0], asset_id + ".json")


def _load_asset(path):
    with open(path, "r", encoding="utf-8") as file:
        asset = json.load(file)
    if not isinstance(asset, dict) or asset.get("format") != FORMAT or int(asset.get("version", 0)) != VERSION:
        raise ValueError(f"参考视频资产格式错误：{os.path.basename(path)}")
    return asset


def _load_assets():
    metadata, _ = _asset_directories()
    assets = []
    for entry in os.scandir(metadata):
        if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
            assets.append(_load_asset(entry.path))
    return sorted(assets, key=lambda item: item.get("created_at", ""), reverse=True)


def _probe(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("未找到 ffprobe，无法导入参考视频资产")
    result = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "无法读取参考视频")
    data = json.loads(result.stdout)
    video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), None)
    duration = float(data.get("format", {}).get("duration", 0))
    if not video or duration <= 0:
        raise ValueError("文件中没有可用的视频流")
    return duration, int(video.get("width", 0)), int(video.get("height", 0))


def _preprocess(source, target, start, end):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法预处理参考视频资产")
    source_duration, width, height = _probe(source)
    finish = source_duration if end <= 0 else min(end, source_duration)
    if start < 0 or start >= finish:
        raise ValueError("截取开始时间必须早于视频结束时间")
    duration = finish - start
    if duration > 15.0 + 1e-6:
        raise ValueError("MiniMax H3 单个参考视频资产不能超过 15 秒")
    command = [ffmpeg, "-y", "-ss", f"{start:.6f}", "-i", source, "-t", f"{duration:.6f}",
        "-map", "0:v:0", "-map", "0:a?", "-vf", f"fps={FPS}", "-c:v", "libx264", "-preset", "medium",
        "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", target]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-2000:] or "参考视频预处理失败")
    processed_duration, processed_width, processed_height = _probe(target)
    return processed_duration, processed_width or width, processed_height or height


def _write_asset(path, asset):
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + "-", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(asset, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


async def get_reference_assets(request):
    try:
        return web.json_response({"success": True, "assets": _load_assets()})
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return web.json_response({"success": False, "error": str(error)}, status=500)


async def import_reference_asset(request):
    metadata, media = _asset_directories()
    asset_id = str(uuid.uuid4())
    source_path = os.path.join(media, asset_id + ".upload")
    target_path = os.path.join(media, asset_id + ".mp4")
    try:
        reader = await request.multipart()
        name = ""
        start = 0.0
        end = 0.0
        source_name = ""
        size = 0
        async for part in reader:
            if part.name == "video":
                source_name = os.path.basename(part.filename or "reference-video")
                with open(source_path, "wb") as file:
                    while chunk := await part.read_chunk(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise ValueError("参考视频文件不能超过 1 GB")
                        file.write(chunk)
            elif part.name == "name":
                name = (await part.text()).strip()
            elif part.name == "trim_start":
                start = float(await part.text() or 0)
            elif part.name == "trim_end":
                end = float(await part.text() or 0)
        if not size:
            raise ValueError("没有收到参考视频文件")
        duration, width, height = await asyncio.to_thread(_preprocess, source_path, target_path, start, end)
        asset = {
            "format": FORMAT,
            "version": VERSION,
            "id": asset_id,
            "display_name": name or os.path.splitext(source_name)[0] or "参考视频",
            "source_filename": source_name,
            "file": {"type": "input", "subfolder": "minimax_h3/reference_videos", "filename": asset_id + ".mp4"},
            "preprocess": {"trim_start": start, "trim_end": end, "fps": FPS},
            "duration": duration,
            "width": width,
            "height": height,
        }
        _write_asset(os.path.join(metadata, asset_id + ".json"), asset)
        return web.json_response({"success": True, "asset": asset})
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        if os.path.exists(target_path):
            os.unlink(target_path)
        return web.json_response({"success": False, "error": str(error)}, status=400)
    finally:
        if os.path.exists(source_path):
            os.unlink(source_path)


async def delete_reference_asset(request):
    try:
        path = _asset_path(request.match_info["asset_id"])
        if os.path.exists(path):
            os.unlink(path)
        return web.json_response({"success": True})
    except (OSError, ValueError) as error:
        return web.json_response({"success": False, "error": str(error)}, status=400)


def register_routes():
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    server.routes.get("/minimax-h3/reference-assets")(get_reference_assets)
    server.routes.post("/minimax-h3/reference-assets")(import_reference_asset)
    server.routes.delete("/minimax-h3/reference-assets/{asset_id}")(delete_reference_asset)


register_routes()
