import hashlib
import json
import os
import re
from pathlib import Path

import folder_paths
import torch
import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import InputImpl, Types, io, ui

from .segments import (_compile_generation_segment, _segment_frame_plan, _segment_ranges,
    _segment_result)
from .utils import _video_size


CACHE_SUBFOLDER = "video/MiniMax H3 Segments"
_CACHE_NAME = re.compile(r"segment_\d{3}_[0-9a-f]{24}\.mp4")


def _cache_path(filename):
    if not _CACHE_NAME.fullmatch(filename):
        raise ValueError("无效的 MiniMax H3 分段缓存文件名")
    root = Path(folder_paths.get_output_directory()).resolve()
    folder = (root / CACHE_SUBFOLDER).resolve()
    if folder != root and root not in folder.parents:
        raise ValueError("MiniMax H3 分段缓存目录超出输出目录")
    return folder / filename


def _latent_path(filename):
    return _cache_path(filename).with_suffix(".safetensors")


def _save_latent(path, latent):
    streams = list(latent["samples"].unbind())
    if len(streams) != 2:
        raise ValueError("分段缓存需要 MiniMax H3 配对的视频和音频 Latent")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    try:
        comfy.utils.save_torch_file({
            "video": streams[0].detach().to(device="cpu").contiguous(),
            "audio": streams[1].detach().to(device="cpu").contiguous(),
        }, temporary, metadata={"format": "minimax-h3-av-latent-v1"})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_latent(path):
    if not path.is_file():
        return None
    tensors = comfy.utils.load_torch_file(str(path), safe_load=True)
    if "video" not in tensors or "audio" not in tensors:
        raise ValueError(f"MiniMax H3 分段 Latent 缓存内容不完整：{path.name}")
    return {"samples": comfy.nested_tensor.NestedTensor((tensors["video"], tensors["audio"]))}


def _saved_results(filenames):
    return [ui.SavedResult(name, CACHE_SUBFOLDER, io.FolderType.output) for name in filenames]


def _tensor_digest(tensor, memo):
    if tensor is None:
        return "none"
    key = id(tensor)
    if key in memo:
        return memo[key]
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    raw = value.view(torch.uint8).numpy()
    digest.update(memoryview(raw).cast("B"))
    memo[key] = digest.hexdigest()
    return memo[key]


def _audio_digest(audio, memo):
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return "none"
    return f"{audio.get('sample_rate')}:{_tensor_digest(audio['waveform'], memo)}"


def segment_cache_files(generation_job, model, sampler, cache_version):
    ranges = _segment_ranges(generation_job.timeline)
    memo = {}
    previous = ""
    files = []
    model_signature = {
        "class": f"{type(model.model).__module__}.{type(model.model).__qualname__}",
        "size": model.model_size(),
        "patch_keys": sorted(map(str, model.patches)),
        "sampler": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "cache_version": int(cache_version),
    }
    for index, (start, end) in enumerate(ranges):
        rendered = _segment_result(generation_job.timeline, start, end)
        if rendered is not None:
            width, height = _video_size(generation_job.megapixels, generation_job.aspect_ratio)
            payload = {"pipeline": "rendered-result-v1", "index": index, "range": [start, end],
                "width": width, "height": height, "version": rendered[1]}
            previous = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            files.append(f"segment_{index + 1:03d}_{previous[:24]}.mp4")
            continue
        available = round((ranges[index - 1][1] - ranges[index - 1][0]) * 24) if index else 0
        frame_plan = _segment_frame_plan(generation_job.continuity_seconds, available,
            round((end - start) * 24))
        compiled, motion_references, standalone_audios = _compile_generation_segment(generation_job, index, frame_plan)
        payload = {
            "pipeline": "latent-locked-dynamic-output-v4",
            "index": index,
            "range": [start, end],
            "prompt": compiled.text,
            "width": compiled.video_settings.width,
            "height": compiled.video_settings.height,
            "length": compiled.video_settings.length,
            "seed": (generation_job.seed + index) & 0xffffffffffffffff,
            "scheduler": generation_job.scheduler,
            "steps": generation_job.steps,
            "denoise": generation_job.denoise,
            "ref_image_size": generation_job.ref_image_size,
            "requested_frames": frame_plan.requested_frames,
            "current_frames": frame_plan.current_frames,
            "locked_frames": frame_plan.locked_frames,
            "model": model_signature,
            "previous": previous,
            "images": [_tensor_digest(item.image, memo) for item in compiled.references],
            "motion": [{
                "role": reference.role,
                "frames": _tensor_digest(reference.frames, memo),
                "audio": _audio_digest(reference.audio, memo),
            } for reference in motion_references],
            "reference_audio": [_audio_digest(audio, memo) for audio in standalone_audios],
        }
        previous = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        files.append(f"segment_{index + 1:03d}_{previous[:24]}.mp4")
    return tuple(files)


class MiniMaxH3SegmentCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentCheckpoint", display_name="MiniMax H3 保存分段（内部）",
            category="MiniMax H3/提示词构建/内部", inputs=[io.Video.Input("video"),
                io.String.Input("cache_file"), io.String.Input("preview_files"),
                io.Latent.Input("latent", optional=True)],
            outputs=[io.Video.Output(display_name="video"), io.Latent.Output(display_name="latent")],
            is_output_node=True)

    @classmethod
    def execute(cls, video, cache_file, preview_files, latent=None):
        path = _cache_path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if latent is not None:
            _save_latent(_latent_path(cache_file), latent)
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mp4")
        try:
            video.save_to(temporary, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        files = json.loads(preview_files)
        saved_video = InputImpl.VideoFromFile(str(path))
        return io.NodeOutput(saved_video, latent, ui=ui.PreviewVideo(_saved_results(files)))


class MiniMaxH3SegmentCheckpointLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentCheckpointLoad", display_name="MiniMax H3 读取分段（内部）",
            category="MiniMax H3/提示词构建/内部", inputs=[io.String.Input("cache_file"),
                io.String.Input("preview_files")], outputs=[io.Video.Output(display_name="video"),
                io.Latent.Output(display_name="latent")])

    @classmethod
    def execute(cls, cache_file, preview_files):
        path = _cache_path(cache_file)
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax H3 分段缓存不存在：{path.name}")
        files = json.loads(preview_files)
        video = InputImpl.VideoFromFile(str(path))
        return io.NodeOutput(video, _load_latent(_latent_path(cache_file)), ui=ui.PreviewVideo(_saved_results(files)))
