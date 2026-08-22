import hashlib
import json
import os
import re
from base64 import b64encode
from io import BytesIO
from pathlib import Path

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import folder_paths
import latent_preview
import torch
from comfy_api.latest import InputImpl, Types, io, ui
from server import PromptServer

from .segments import (_compile_generation_segment, _context_frame_count, _segment_ranges,
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
        context_frames = _context_frame_count(generation_job.continuity_seconds, available)
        compiled, motion_references = _compile_generation_segment(generation_job, index, context_frames)
        payload = {
            "pipeline": "locked-av-context-v1",
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
            "context_frames": context_frames,
            "model": model_signature,
            "previous": previous,
            "images": [_tensor_digest(item.image, memo) for item in compiled.references],
            "motion": [{
                "role": reference.role,
                "frames": _tensor_digest(reference.frames, memo),
                "audio": _audio_digest(reference.audio, memo),
            } for reference in motion_references],
        }
        previous = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        files.append(f"segment_{index + 1:03d}_{previous[:24]}.mp4")
    return tuple(files)


def _latent_video_frames(previewer, video):
    return [previewer.decode_latent_to_preview(video[:, :, index:index + 1])
        for index in range(int(video.shape[2]))]


def _jpeg_b64(frame):
    buffer = BytesIO()
    frame.convert("RGB").save(buffer, format="JPEG", quality=72)
    return b64encode(buffer.getvalue()).decode("ascii")


def _send_step_video_preview(node_id, segment_index, segment_total, step, total_steps, duration_seconds, frames):
    if not node_id or not frames:
        return
    server = PromptServer.instance
    server.send_sync("minimax_h3_step_video_preview", {
        "node_id": str(node_id),
        "segment_index": int(segment_index),
        "segment_total": int(segment_total),
        "step": int(step),
        "total_steps": int(total_steps),
        "duration_seconds": float(duration_seconds),
        "frames": [_jpeg_b64(frame) for frame in frames],
    }, server.client_id)


class MiniMaxH3SegmentCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentCheckpoint", display_name="MiniMax H3 保存分段（内部）",
            category="MiniMax H3/提示词构建/内部", inputs=[io.Video.Input("video"),
                io.String.Input("cache_file"), io.String.Input("preview_files")],
            outputs=[io.Video.Output(display_name="video")], is_output_node=True)

    @classmethod
    def execute(cls, video, cache_file, preview_files):
        path = _cache_path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mp4")
        try:
            video.save_to(temporary, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        files = json.loads(preview_files)
        saved_video = InputImpl.VideoFromFile(str(path))
        return io.NodeOutput(saved_video, ui=ui.PreviewVideo(_saved_results(files)))


class MiniMaxH3SegmentCheckpointLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentCheckpointLoad", display_name="MiniMax H3 读取分段（内部）",
            category="MiniMax H3/提示词构建/内部", inputs=[io.String.Input("cache_file"),
                io.String.Input("preview_files")], outputs=[io.Video.Output(display_name="video")])

    @classmethod
    def execute(cls, cache_file, preview_files):
        path = _cache_path(cache_file)
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax H3 分段缓存不存在：{path.name}")
        files = json.loads(preview_files)
        video = InputImpl.VideoFromFile(str(path))
        return io.NodeOutput(video, ui=ui.PreviewVideo(_saved_results(files)))


class MiniMaxH3PreviewSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PreviewSampler", display_name="MiniMax H3 实时预览采样（内部）",
            category="MiniMax H3/提示词构建/内部", inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"),
                io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"), io.Latent.Input("latent_image"),
                io.Int.Input("preview_every_steps", min=1, max=20, default=1),
                io.String.Input("preview_node_id"), io.Int.Input("segment_index", min=1),
                io.Int.Input("segment_total", min=1),
                io.Float.Input("preview_duration_seconds", min=0.01, step=0.01)],
            outputs=[io.Latent.Output(display_name="output"), io.Latent.Output(display_name="denoised_output")])

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, preview_every_steps, preview_node_id,
                segment_index, segment_total, preview_duration_seconds):
        latent = latent_image.copy()
        samples_in = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent["samples"],
            latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
        latent["samples"] = samples_in
        noise_mask = latent.get("noise_mask")
        x0_output = {}
        latent_format = guider.model_patcher.model.latent_format
        previewer = latent_preview.Latent2RGBPreviewer(latent_format.latent_rgb_factors,
            latent_format.latent_rgb_factors_bias, latent_format.latent_rgb_factors_reshape)
        progress = comfy.utils.ProgressBar(sigmas.shape[-1] - 1)

        def callback(step, x0, _x, total_steps):
            x0_output["x0"] = x0
            if (step + 1) % int(preview_every_steps) == 0 or step + 1 == total_steps:
                video = x0.tensors[0] if x0.is_nested else x0
                frames = _latent_video_frames(previewer, video)
                _send_step_video_preview(preview_node_id, segment_index, segment_total, step + 1,
                    total_steps, preview_duration_seconds, frames)
            progress.update_absolute(step + 1, total_steps)

        samples = guider.sample(noise.generate_noise(latent), samples_in, sampler, sigmas,
            denoise_mask=noise_mask, callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=noise.seed).to(comfy.model_management.intermediate_device())
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        x0 = x0_output.get("x0")
        if x0 is None:
            return io.NodeOutput(out, out)
        if samples.is_nested and not x0.is_nested:
            shapes = [value.shape for value in samples.unbind()]
            x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, shapes))
        denoised = latent.copy()
        denoised["samples"] = guider.model_patcher.model.process_latent_out(x0.cpu())
        return io.NodeOutput(out, denoised)
