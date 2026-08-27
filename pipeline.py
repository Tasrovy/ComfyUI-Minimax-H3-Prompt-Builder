import json

import comfy.model_management
import comfy.sample
import comfy.utils
import folder_paths
import torch

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .schema import (ASPECT_RATIOS, CATEGORY, FPS, H3_GENERATION_JOB, H3_SECOND_PASS_BATCH,
    H3_SECOND_PASS_ENTRY, SecondPassBatchData, SecondPassEntryData)
from .segments import (_lock_context_prefix, _segment_context_mode, _segment_continuity_seconds,
    _segment_frame_plan, _segment_ranges, _segment_result, _segment_visible_frames)
from .checkpoints import (_cache_path, _latent_path, second_pass_cache_files,
    segment_cache_files)
from .utils import _video_size


SECOND_PASS_UPSCALE_METHODS = {
    "Lanczos（高质量）": "lanczos",
    "双三次": "bicubic",
    "双线性": "bilinear",
    "区域平均": "area",
    "最近邻": "nearest-exact",
}
SECOND_PASS_CROP_MODES = {"拉伸到目标比例": "disabled", "居中裁切到目标比例": "center"}
AUDIO_SAMPLE_TOLERANCE = 8


def _latent_upscaler_models():
    try:
        models = folder_paths.get_filename_list("latent_upscale_models")
    except KeyError:
        models = []
    return models or ["请先安装 Minimax H3 Latent Upscaler 3D 并放入模型"]


class MiniMaxH3SegmentSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentSampler", display_name="MiniMax H3 分段采样（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"),
                io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"), io.Latent.Input("latent_image")],
            outputs=[io.Latent.Output(display_name="output")])

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image.copy()
        samples_in = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent["samples"],
            latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
        latent["samples"] = samples_in
        progress = comfy.utils.ProgressBar(sigmas.shape[-1] - 1)

        def callback(step, _x0, _x, total_steps):
            progress.update_absolute(step + 1, total_steps)

        samples = guider.sample(noise.generate_noise(latent), samples_in, sampler, sigmas,
            denoise_mask=latent.get("noise_mask"), callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise.seed)
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples.to(comfy.model_management.intermediate_device())
        return io.NodeOutput(out)


class MiniMaxH3SegmentResultPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentResultPrepare", display_name="MiniMax H3 整理已有片段（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Video.Input("video"),
                io.Float.Input("duration_seconds", min=0.01, step=0.01),
                io.Int.Input("width", min=32), io.Int.Input("height", min=32)],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")])

    @classmethod
    def execute(cls, video, duration_seconds, width, height):
        components = video.get_components()
        frames = components.images[..., :3]
        source_fps = float(components.frame_rate)
        target_frames = max(1, round(duration_seconds * FPS))
        if source_fps <= 0 or frames.shape[0] / source_fps + 0.5 / FPS < duration_seconds:
            raise ValueError("已生成结果视频短于其动作片段，无法作为完整片段使用")
        positions = torch.arange(target_frames, device=frames.device, dtype=torch.float32) * (source_fps / FPS)
        indices = positions.round().long().clamp(max=frames.shape[0] - 1)
        frames = frames.index_select(0, indices)
        if frames.shape[1:3] != (height, width):
            frames = comfy.utils.common_upscale(frames.movedim(-1, 1), width, height,
                "lanczos", "disabled").movedim(1, -1)

        sample_rate = 32000
        sample_count = max(1, round(duration_seconds * sample_rate))
        if components.audio is None:
            waveform = torch.zeros((1, 2, sample_count), dtype=torch.float32)
        else:
            waveform = components.audio["waveform"]
            source_rate = int(components.audio["sample_rate"])
            if source_rate != sample_rate:
                resampled = max(1, round(waveform.shape[-1] * sample_rate / source_rate))
                shape = waveform.shape
                waveform = torch.nn.functional.interpolate(waveform.reshape(-1, 1, shape[-1]),
                    size=resampled, mode="linear", align_corners=False).reshape(*shape[:-1], resampled)
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)
            elif waveform.shape[1] > 2:
                waveform = waveform[:, :2]
            waveform = torch.nn.functional.pad(waveform[..., :sample_count],
                (0, max(0, sample_count - waveform.shape[-1])))
        return io.NodeOutput(frames, {"waveform": waveform, "sample_rate": sample_rate})


class MiniMaxH3SegmentTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentTrim", display_name="MiniMax H3 片段裁切（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Image.Input("images"), io.Audio.Input("audio"),
            io.Int.Input("context_frames", min=0), io.Float.Input("duration_seconds", min=0.01, step=0.01),
            io.Int.Input("generated_frames", min=0, default=0, optional=True),
            io.Int.Input("visible_frames", min=0, default=0, optional=True)],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")])

    @classmethod
    def execute(cls, images, audio, context_frames, duration_seconds, generated_frames=0, visible_frames=0):
        if generated_frames and images.shape[0] != generated_frames:
            raise ValueError(f"模型解码得到 {images.shape[0]} 帧，但本次参考视频与生成目标均为 {generated_frames} 帧")
        context_frames = min(max(0, int(context_frames)), max(0, images.shape[0] - 1))
        images = images[context_frames:]
        frame_count = min(images.shape[0], max(1, visible_frames or round(duration_seconds * FPS)))
        images = images[:frame_count]
        sample_rate = audio["sample_rate"]
        audio_offset = min(audio["waveform"].shape[-1], round((context_frames / FPS) * sample_rate))
        waveform = audio["waveform"][..., audio_offset:]
        sample_count = round((frame_count / FPS) * sample_rate)
        waveform = torch.nn.functional.pad(waveform[..., :sample_count],
            (0, max(0, sample_count - waveform.shape[-1])))
        return io.NodeOutput(images, {**audio, "waveform": waveform})


class MiniMaxH3SecondPassResize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SecondPassResize", display_name="MiniMax H3 二采画面放大（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Image.Input("images"),
                io.Float.Input("megapixels", min=0.01, max=16.0, step=0.01),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS)),
                io.Combo.Input("upscale_method", options=list(SECOND_PASS_UPSCALE_METHODS)),
                io.Combo.Input("crop_mode", options=list(SECOND_PASS_CROP_MODES))],
            outputs=[io.Image.Output(display_name="images")])

    @classmethod
    def execute(cls, images, megapixels, aspect_ratio, upscale_method="Lanczos（高质量）",
                crop_mode="拉伸到目标比例"):
        width, height = _video_size(megapixels, aspect_ratio)
        images = images[..., :3]
        if images.shape[1:3] != (height, width):
            images = comfy.utils.common_upscale(images.movedim(-1, 1), width, height,
                SECOND_PASS_UPSCALE_METHODS[upscale_method], SECOND_PASS_CROP_MODES[crop_mode]).movedim(1, -1)
        return io.NodeOutput(images)


class MiniMaxH3SecondPassEntryPack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SecondPassEntryPack",
            display_name="MiniMax H3 二采片段打包（内部）", category=f"{CATEGORY}/内部",
            inputs=[io.Video.Input("video"), io.Int.Input("segment_index"),
                io.Float.Input("start_time"), io.Float.Input("end_time"),
                io.Int.Input("context_frames"), io.Int.Input("generation_frames"),
                io.Int.Input("visible_frames"), io.Float.Input("visible_duration"), io.String.Input("cache_file"),
                io.Combo.Input("context_mode", options=["full", "audio", "off"], default="full"),
                io.Conditioning.Input("conditioning", optional=True), io.Latent.Input("latent", optional=True)],
            outputs=[H3_SECOND_PASS_ENTRY.Output(display_name="entry")])

    @classmethod
    def execute(cls, video, segment_index, start_time, end_time, context_frames, generation_frames,
                visible_frames, visible_duration, cache_file, context_mode="full", conditioning=None, latent=None):
        return io.NodeOutput(SecondPassEntryData(segment_index, conditioning, latent, video, start_time,
            end_time, context_frames, generation_frames, visible_frames, visible_duration, cache_file, context_mode))


class MiniMaxH3SecondPassBatchAppend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SecondPassBatchAppend",
            display_name="MiniMax H3 二采任务打包（内部）", category=f"{CATEGORY}/内部",
            inputs=[H3_SECOND_PASS_ENTRY.Input("entry"), H3_SECOND_PASS_BATCH.Input("batch", optional=True)],
            outputs=[H3_SECOND_PASS_BATCH.Output(display_name="second_pass_batch")])

    @classmethod
    def execute(cls, entry, batch=None):
        entries = batch.entries if batch is not None else ()
        return io.NodeOutput(SecondPassBatchData((*entries, entry)))


class MiniMaxH3SecondPassBatchParser(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SecondPassBatchParser",
            display_name="MiniMax H3 二采任务解析（Second Pass Batch Parser）", category=CATEGORY,
            description="将多段生成节点的二采任务包拆成逐片段列表，供外部放大、重编码或采样节点使用。",
            inputs=[H3_SECOND_PASS_BATCH.Input("second_pass_batch")],
            outputs=[io.Latent.Output(display_name="first_pass_latent", is_output_list=True),
                io.Conditioning.Output(display_name="conditioning", is_output_list=True),
                io.Video.Output(display_name="source_video", is_output_list=True),
                io.Int.Output(display_name="segment_index", is_output_list=True),
                io.Int.Output(display_name="context_frames", is_output_list=True),
                io.Int.Output(display_name="generation_frames", is_output_list=True),
                io.Float.Output(display_name="visible_duration", is_output_list=True),
                io.String.Output(display_name="cache_file", is_output_list=True),
                io.Float.Output(display_name="start_time", is_output_list=True),
                io.Float.Output(display_name="end_time", is_output_list=True),
                io.Int.Output(display_name="visible_frames", is_output_list=True)])

    @classmethod
    def execute(cls, second_pass_batch):
        if not isinstance(second_pass_batch, SecondPassBatchData):
            raise TypeError("二采任务解析需要 MiniMax H3 二采任务包")
        entries = tuple(entry for entry in second_pass_batch.entries
            if entry.latent is not None and entry.conditioning is not None)
        if not entries:
            raise ValueError("二采任务包中没有可采样片段；直接导入的已有结果不包含 Latent 和 Conditioning")
        return io.NodeOutput(
            [entry.latent for entry in entries],
            [entry.conditioning for entry in entries],
            [entry.video for entry in entries],
            [entry.segment_index + 1 for entry in entries],
            [entry.context_frames for entry in entries],
            [entry.generation_frames for entry in entries],
            [entry.visible_duration for entry in entries],
            [entry.cache_file for entry in entries],
            [entry.start_time for entry in entries],
            [entry.end_time for entry in entries],
            [entry.visible_frames for entry in entries],
        )


class MiniMaxH3SecondPassLock(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SecondPassLock",
            display_name="MiniMax H3 二采段间锁定（内部）", category=f"{CATEGORY}/内部",
            inputs=[io.Latent.Input("latent"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
                io.Int.Input("locked_frames"),
                io.Boolean.Input("lock_video", default=True),
                io.Int.Input("width", default=0, min=0, optional=True),
                io.Int.Input("height", default=0, min=0, optional=True),
                io.Image.Input("previous_images", optional=True), io.Audio.Input("previous_audio", optional=True),
                io.Latent.Input("previous_latent", optional=True)],
            outputs=[io.Latent.Output(display_name="latent")])

    @classmethod
    def execute(cls, latent, video_vae, audio_vae, locked_frames, lock_video=True, width=0, height=0, previous_images=None,
                previous_audio=None, previous_latent=None):
        if not width or not height:
            video = latent["samples"][0]
            ratio = int(video_vae.spacial_compression_encode())
            height = video.shape[-2] * ratio
            width = video.shape[-1] * ratio
        return io.NodeOutput(_lock_context_prefix(latent, previous_images, previous_audio, previous_latent,
            video_vae, audio_vae, locked_frames, width, height, lock_video=lock_video))


class MiniMaxH3SecondPassUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SecondPassUpscale",
            display_name="MiniMax H3 二次采样放大（Second Pass Upscale）", category=CATEGORY,
            description="将第一次采样的画面潜变量解码放大并重新编码，保留原音频潜变量，再以较低降噪强度进行第二次采样。",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("conditioning"),
                io.Latent.Input("first_pass_latent"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
                io.Sampler.Input("sampler", optional=True,
                    tooltip="可选。连接后覆盖节点内的“二采采样器”选择。"),
                io.Float.Input("megapixels", display_name="二采百万像素", default=1.0,
                    min=0.01, max=16.0, step=0.01),
                io.Combo.Input("aspect_ratio", display_name="二采宽高比",
                    options=list(ASPECT_RATIOS), default="16:9"),
                io.Combo.Input("upscale_method", display_name="画面放大算法",
                    options=list(SECOND_PASS_UPSCALE_METHODS), default="Lanczos（高质量）"),
                io.Combo.Input("crop_mode", display_name="宽高比处理",
                    options=list(SECOND_PASS_CROP_MODES), default="拉伸到目标比例"),
                io.Combo.Input("sampler_name", display_name="二采采样器",
                    options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                io.Int.Input("seed", display_name="二采噪声种子", default=0, min=0,
                    max=0xffffffffffffffff, control_after_generate=True),
                io.Combo.Input("scheduler", display_name="二采调度器",
                    options=comfy.samplers.SCHEDULER_NAMES, default="beta"),
                io.Int.Input("steps", display_name="二采采样步数", default=4, min=1, max=10000),
                io.Float.Input("denoise", display_name="二采降噪强度", default=0.2,
                    min=0.0, max=1.0, step=0.01)],
            outputs=[io.Video.Output(display_name="video")], enable_expand=True)

    @classmethod
    def execute(cls, model, conditioning, first_pass_latent, video_vae, audio_vae, sampler=None,
                megapixels=1.0, aspect_ratio="16:9", upscale_method="Lanczos（高质量）",
                crop_mode="拉伸到目标比例", sampler_name="euler", seed=0, scheduler="beta", steps=4,
                denoise=0.2):
        graph = GraphBuilder()
        parent_node_id = cls.hidden.unique_id if cls.hidden is not None else None

        def stage_node(class_type, node_id, **inputs):
            node = graph.node(class_type, id=node_id, **inputs)
            if parent_node_id is not None:
                node.set_override_display_id(parent_node_id)
            return node

        separated = stage_node("LTXVSeparateAVLatent", "second_pass_separate", av_latent=first_pass_latent)
        decoded = stage_node("VAEDecode", "second_pass_decode_source",
            samples=separated.out(0), vae=video_vae)
        resized = stage_node("MiniMaxH3SecondPassResize", "second_pass_resize", images=decoded.out(0),
            megapixels=megapixels, aspect_ratio=aspect_ratio, upscale_method=upscale_method,
            crop_mode=crop_mode)
        video_latent = stage_node("VAEEncode", "second_pass_encode_video",
            pixels=resized.out(0), vae=video_vae)
        av_latent = stage_node("LTXVConcatAVLatent", "second_pass_concat",
            video_latent=video_latent.out(0), audio_latent=separated.out(1))
        noise = stage_node("RandomNoise", "second_pass_noise", noise_seed=seed)
        guider = stage_node("BasicGuider", "second_pass_guider", model=model, conditioning=conditioning)
        sigmas = stage_node("BasicScheduler", "second_pass_scheduler", model=model,
            scheduler=scheduler, steps=steps, denoise=denoise)
        if sampler is None:
            sampler = stage_node("KSamplerSelect", "second_pass_sampler", sampler_name=sampler_name).out(0)
        sampled = stage_node("SamplerCustomAdvanced", "second_pass_sampling", noise=noise.out(0),
            guider=guider.out(0), sampler=sampler, sigmas=sigmas.out(0), latent_image=av_latent.out(0))
        images = stage_node("VAEDecode", "second_pass_decode_video",
            samples=sampled.out(0), vae=video_vae)
        audio = stage_node("VAEDecodeAudio", "second_pass_decode_audio",
            samples=sampled.out(0), vae=audio_vae)
        video = stage_node("CreateVideo", "second_pass_video", images=images.out(0),
            fps=float(FPS), audio=audio.out(0))
        return io.NodeOutput(video.out(0), expand=graph.finalize())


class MiniMaxH3SegmentJoin(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentJoin", 
            display_name="MiniMax H3 分段拼接（内部）",
            category=f"{CATEGORY}/内部", 
            inputs=[
                io.Image.Input("previous_images"), 
                io.Audio.Input("previous_audio"),
                io.Image.Input("current_images"), 
                io.Audio.Input("current_audio")
            ],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")]
        )

    @classmethod
    def execute(cls, previous_images, previous_audio, current_images, current_audio):
        if previous_images.shape[1:] != current_images.shape[1:]:
            raise ValueError("分段画面尺寸不一致，无法拼接")
        if previous_audio["sample_rate"] != current_audio["sample_rate"]:
            raise ValueError("分段音频采样率不一致，无法拼接")
        sample_rate = previous_audio["sample_rate"]
        previous_samples = round((previous_images.shape[0] / FPS) * sample_rate)
        current_samples = round((current_images.shape[0] / FPS) * sample_rate)
        previous_waveform = previous_audio["waveform"]
        current_waveform = current_audio["waveform"]
        if abs(previous_waveform.shape[-1] - previous_samples) > AUDIO_SAMPLE_TOLERANCE:
            raise ValueError("上一段音频长度与画面帧数不一致，无法拼接")
        if abs(current_waveform.shape[-1] - current_samples) > AUDIO_SAMPLE_TOLERANCE:
            raise ValueError("当前段音频长度与画面帧数不一致，无法拼接")

        images = torch.cat((previous_images, current_images), dim=0)
        waveform = torch.cat((previous_waveform, current_waveform), dim=-1)
        sample_count = round((images.shape[0] / FPS) * sample_rate)
        waveform = torch.nn.functional.pad(waveform[..., :sample_count],
            (0, max(0, sample_count - waveform.shape[-1])))
        return io.NodeOutput(images, {**current_audio, "waveform": waveform})


class MiniMaxH3MultiSegmentSecondPass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MultiSegmentSecondPass",
            display_name="MiniMax H3 多段二次采样（Decoded Video）", category=CATEGORY,
            description="接收一采节点输出的二采任务包，逐段放大、低降噪二采、保存未裁剪片段并预览。",
            inputs=[io.Model.Input("model"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
                io.Sampler.Input("sampler"), H3_SECOND_PASS_BATCH.Input("second_pass_batch"),
                io.Float.Input("megapixels", display_name="二采百万像素", default=1.0,
                    min=0.01, max=16.0, step=0.01),
                io.Combo.Input("aspect_ratio", display_name="二采宽高比",
                    options=list(ASPECT_RATIOS), default="16:9"),
                io.Combo.Input("upscale_method", display_name="画面放大算法",
                    options=list(SECOND_PASS_UPSCALE_METHODS), default="Lanczos（高质量）"),
                io.Combo.Input("crop_mode", display_name="宽高比处理",
                    options=list(SECOND_PASS_CROP_MODES), default="拉伸到目标比例"),
                io.Int.Input("seed", display_name="二采噪声种子", default=0, min=0,
                    max=0xffffffffffffffff, control_after_generate=True),
                io.Combo.Input("scheduler", display_name="二采调度器",
                    options=comfy.samplers.SCHEDULER_NAMES, default="beta"),
                io.Int.Input("steps", display_name="二采采样步数", default=4, min=1, max=10000),
                io.Float.Input("denoise", display_name="二采降噪强度", default=0.2,
                    min=0.0, max=1.0, step=0.01),
                io.Combo.Input("cache_mode", display_name="二采分段缓存",
                    options=["复用已完成片段", "重新生成全部片段"], default="复用已完成片段"),
                io.Int.Input("cache_version", display_name="二采缓存版本", default=0, min=0, max=1000000)],
            outputs=[io.Video.Output(display_name="video")], enable_expand=True)

    @classmethod
    def execute(cls, model, video_vae, audio_vae, sampler, second_pass_batch, megapixels=1.0,
                aspect_ratio="16:9", upscale_method="Lanczos（高质量）", crop_mode="拉伸到目标比例",
                seed=0, scheduler="beta", steps=4, denoise=0.2, cache_mode="复用已完成片段",
                cache_version=0, _latent_upscale=None, _sigmas=None, _parent_node_id=None):
        if not isinstance(second_pass_batch, SecondPassBatchData) or not second_pass_batch.entries:
            raise ValueError("二采任务包为空")
        settings = {"megapixels": megapixels, "aspect_ratio": aspect_ratio,
            "upscale_method": upscale_method, "crop_mode": crop_mode, "seed": seed,
            "scheduler": scheduler, "steps": steps, "denoise": denoise}
        if _latent_upscale is not None:
            settings["latent_upscale"] = _latent_upscale
        if _sigmas is not None:
            settings["sigmas"] = _sigmas.detach().to(device="cpu", dtype=torch.float32).tolist()
        cache_files = second_pass_cache_files(second_pass_batch, model, sampler, settings, cache_version)
        width, height = _video_size(megapixels, aspect_ratio) if _latent_upscale is None else (0, 0)
        graph = GraphBuilder()
        parent_node_id = (_parent_node_id if _parent_node_id is not None else
            (cls.hidden.unique_id if cls.hidden is not None else None))

        def stage_node(class_type, node_id, **inputs):
            node = graph.node(class_type, id=node_id, **inputs)
            if parent_node_id is not None:
                node.set_override_display_id(parent_node_id)
            return node

        def upscale_latent(latent, node_id):
            return stage_node("MinimaxH3LatentUpscaler3D", node_id, latent=latent,
                model_name=_latent_upscale["model_name"],
                mode="megapixels", **{"mode.megapixels": megapixels},
                align=_latent_upscale["align"], enable_chunking=_latent_upscale["enable_chunking"],
                device=_latent_upscale["device"], precision=_latent_upscale["precision"]).out(0)

        accumulated_images = None
        accumulated_audio = None
        previous_images = None
        previous_audio = None
        previous_latent = None
        total = len(second_pass_batch.entries)
        for position, entry in enumerate(second_pass_batch.entries):
            stage = f"segment_{position + 1}_of_{total}"
            preview_files = json.dumps(cache_files[:position + 1])
            checkpoint_latent = None
            if cache_mode == "复用已完成片段" and _cache_path(cache_files[position]).is_file():
                checkpoint = stage_node("MiniMaxH3SegmentCheckpointLoad", f"{stage}_cache_load",
                    cache_file=cache_files[position], preview_files=preview_files)
                if _latent_path(cache_files[position]).is_file():
                    checkpoint_latent = checkpoint.out(1)
            elif entry.latent is None or entry.conditioning is None:
                source = stage_node("GetVideoComponents", f"{stage}_source_components", video=entry.video)
                if _latent_upscale is None:
                    resized_images = stage_node("MiniMaxH3SecondPassResize", f"{stage}_resize",
                        images=source.out(0), megapixels=megapixels, aspect_ratio=aspect_ratio,
                        upscale_method=upscale_method, crop_mode=crop_mode).out(0)
                else:
                    source_latent = stage_node("VAEEncode", f"{stage}_encode",
                        pixels=source.out(0), vae=video_vae)
                    upscaled = upscale_latent(source_latent.out(0), f"{stage}_resize")
                    resized_images = stage_node("VAEDecode", f"{stage}_video_decode",
                        samples=upscaled, vae=video_vae).out(0)
                segment_video = stage_node("CreateVideo", f"{stage}_segment_video",
                    images=resized_images, fps=float(FPS), audio=source.out(1))
                checkpoint = stage_node("MiniMaxH3SegmentCheckpoint", f"{stage}_checkpoint",
                    video=segment_video.out(0), cache_file=cache_files[position], preview_files=preview_files)
            else:
                separated = stage_node("LTXVSeparateAVLatent", f"{stage}_separate", av_latent=entry.latent)
                if _latent_upscale is None:
                    decoded = stage_node("VAEDecode", f"{stage}_source_decode",
                        samples=separated.out(0), vae=video_vae)
                    resized = stage_node("MiniMaxH3SecondPassResize", f"{stage}_resize", images=decoded.out(0),
                        megapixels=megapixels, aspect_ratio=aspect_ratio, upscale_method=upscale_method,
                        crop_mode=crop_mode)
                    video_latent = stage_node("VAEEncode", f"{stage}_encode",
                        pixels=resized.out(0), vae=video_vae).out(0)
                else:
                    video_latent = upscale_latent(separated.out(0), f"{stage}_resize")
                av_latent = stage_node("LTXVConcatAVLatent", f"{stage}_concat",
                    video_latent=video_latent, audio_latent=separated.out(1))
                latent = av_latent.out(0)
                if entry.context_frames and previous_audio is not None:
                    lock_inputs = {"latent": latent, "video_vae": video_vae, "audio_vae": audio_vae,
                        "locked_frames": entry.context_frames, "width": width, "height": height,
                        "lock_video": entry.context_mode == "full", "previous_images": previous_images,
                        "previous_audio": previous_audio}
                    if previous_latent is not None:
                        lock_inputs["previous_latent"] = previous_latent
                    latent = stage_node("MiniMaxH3SecondPassLock", f"{stage}_lock", **lock_inputs).out(0)
                noise = stage_node("RandomNoise", f"{stage}_noise",
                    noise_seed=(seed + entry.segment_index) & 0xffffffffffffffff)
                guider = stage_node("BasicGuider", f"{stage}_guider",
                    model=model, conditioning=entry.conditioning)
                sigmas = (_sigmas if _sigmas is not None else
                    stage_node("BasicScheduler", f"{stage}_scheduler", model=model,
                        scheduler=scheduler, steps=steps, denoise=denoise).out(0))
                sampled = stage_node("MiniMaxH3SegmentSampler", f"{stage}_sampling", noise=noise.out(0),
                    guider=guider.out(0), sampler=sampler, sigmas=sigmas, latent_image=latent)
                decoded_images = stage_node("VAEDecode", f"{stage}_video_decode",
                    samples=sampled.out(0), vae=video_vae).out(0)
                decoded_audio = stage_node("VAEDecodeAudio", f"{stage}_audio_decode",
                    samples=sampled.out(0), vae=audio_vae).out(0)
                segment_video = stage_node("CreateVideo", f"{stage}_segment_video",
                    images=decoded_images, fps=float(FPS), audio=decoded_audio)
                checkpoint = stage_node("MiniMaxH3SegmentCheckpoint", f"{stage}_checkpoint",
                    video=segment_video.out(0), latent=sampled.out(0), cache_file=cache_files[position],
                    preview_files=preview_files)
                checkpoint_latent = sampled.out(0)

            components = stage_node("GetVideoComponents", f"{stage}_components", video=checkpoint.out(0))
            images = components.out(0)
            audio = components.out(1)
            trimmed = stage_node("MiniMaxH3SegmentTrim", f"{stage}_trim", images=images,
                audio=audio, context_frames=entry.context_frames,
                duration_seconds=entry.visible_duration, generated_frames=entry.generation_frames,
                visible_frames=entry.visible_frames)
            images = trimmed.out(0)
            audio = trimmed.out(1)
            previous_images = images
            previous_audio = audio
            previous_latent = checkpoint_latent
            if accumulated_images is None:
                accumulated_images = images
                accumulated_audio = audio
            else:
                joined = stage_node("MiniMaxH3SegmentJoin", f"{stage}_join",
                    previous_images=accumulated_images, previous_audio=accumulated_audio,
                    current_images=images, current_audio=audio)
                accumulated_images = joined.out(0)
                accumulated_audio = joined.out(1)

        video = stage_node("CreateVideo", f"segment_{total}_of_{total}_final_video",
            images=accumulated_images, fps=float(FPS), audio=accumulated_audio)
        return io.NodeOutput(video.out(0), expand=graph.finalize())


class MiniMaxH3MultiSegmentLatentSecondPass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MultiSegmentLatentSecondPass",
            display_name="MiniMax H3 多段 Latent 模型二采（Decoded Video）", category=CATEGORY,
            description="使用 Minimax H3 3D Latent 放大模型逐段升分辨率、二采、缓存，并按任务包自动裁剪和拼接。",
            inputs=[io.Model.Input("model"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
                io.Sampler.Input("sampler"), H3_SECOND_PASS_BATCH.Input("second_pass_batch"),
                io.Combo.Input("upscaler_model", display_name="Latent 放大模型",
                    options=_latent_upscaler_models()),
                io.Float.Input("megapixels", display_name="目标百万像素", default=1.0,
                    min=0.1, max=8.0, step=0.1),
                io.Int.Input("align", display_name="尺寸对齐", default=32, min=1, max=512),
                io.Boolean.Input("enable_chunking", display_name="时间分块", default=True),
                io.Combo.Input("device", display_name="放大设备",
                    options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("precision", display_name="放大精度",
                    options=["fp32", "fp16", "bf16"], default="bf16"),
                io.Sigmas.Input("sigmas", display_name="二采 Sigmas", optional=True),
                io.Int.Input("seed", display_name="二采噪声种子", default=0, min=0,
                    max=0xffffffffffffffff, control_after_generate=True),
                io.Combo.Input("scheduler", display_name="未连接 Sigmas 时的调度器",
                    options=comfy.samplers.SCHEDULER_NAMES, default="beta"),
                io.Int.Input("steps", display_name="未连接 Sigmas 时的步数", default=4, min=1, max=10000),
                io.Float.Input("denoise", display_name="未连接 Sigmas 时的降噪强度", default=0.2,
                    min=0.0, max=1.0, step=0.01),
                io.Combo.Input("cache_mode", display_name="二采分段缓存",
                    options=["复用已完成片段", "重新生成全部片段"], default="复用已完成片段"),
                io.Int.Input("cache_version", display_name="二采缓存版本", default=0,
                    min=0, max=1000000)],
            outputs=[io.Video.Output(display_name="video")], enable_expand=True)

    @classmethod
    def execute(cls, model, video_vae, audio_vae, sampler, second_pass_batch, upscaler_model,
                megapixels=1.0, align=32, enable_chunking=True, device="cuda", precision="bf16",
                sigmas=None, seed=0, scheduler="beta", steps=4, denoise=0.2,
                cache_mode="复用已完成片段", cache_version=0):
        if upscaler_model.startswith("请先安装"):
            raise ValueError(upscaler_model)
        latent_upscale = {"model_name": upscaler_model, "align": align,
            "enable_chunking": enable_chunking, "device": device, "precision": precision}
        parent_node_id = cls.hidden.unique_id if cls.hidden is not None else None
        return MiniMaxH3MultiSegmentSecondPass.execute(model, video_vae, audio_vae, sampler,
            second_pass_batch, megapixels=megapixels, seed=seed, scheduler=scheduler, steps=steps,
            denoise=denoise, cache_mode=cache_mode, cache_version=cache_version,
            _latent_upscale=latent_upscale, _sigmas=sigmas, _parent_node_id=parent_node_id)


class MiniMaxH3MultiSegmentGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiSegmentGenerate", 
            display_name="MiniMax H3 多段生成（Decoded Video）",
            category=CATEGORY, 
            description="自动逐段生成、保存未裁剪原始片段并拼接；停止后保留已完成片段，再次运行可从缓存继续。",
            inputs=[
                io.Model.Input("model"), 
                io.Clip.Input("clip"), 
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"), 
                io.Sampler.Input("sampler"), 
                H3_GENERATION_JOB.Input("generation_job"),
                io.Combo.Input("cache_mode", display_name="分段缓存", options=["复用已完成片段", "重新生成全部片段"],
                    default="复用已完成片段"),
                io.Int.Input("cache_version", display_name="缓存版本", default=0, min=0, max=1000000,
                    tooltip="更换模型、CLIP、VAE 或采样器设置后递增此值，以免复用旧模型生成的片段。")
            ],
            outputs=[io.Video.Output(display_name="video"),
                H3_SECOND_PASS_BATCH.Output(display_name="second_pass_batch")],
            enable_expand=True
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, sampler, generation_job,
                cache_mode="复用已完成片段", cache_version=0):
        ranges = _segment_ranges(generation_job.timeline)
        cache_files = segment_cache_files(generation_job, model, sampler, cache_version)
        graph = GraphBuilder()
        parent_node_id = cls.hidden.unique_id if cls.hidden is not None else None

        def stage_node(class_type, node_id, **inputs):
            node = graph.node(class_type, id=node_id, **inputs)
            if parent_node_id is not None:
                node.set_override_display_id(parent_node_id)
            return node

        accumulated_images = None
        accumulated_audio = None
        previous_images = None
        previous_audio = None
        previous_latent = None
        second_pass_batch = None
        
        for index in range(len(ranges)):
            stage = f"segment_{index + 1}_of_{len(ranges)}"
            preview_files = json.dumps(cache_files[:index + 1])
            start, end = ranges[index]
            segment_duration = end - start
            segment_result = _segment_result(generation_job.timeline, start, end)
            available_frames = _segment_visible_frames(*ranges[index - 1]) if index else 0
            frame_plan = (_segment_frame_plan(_segment_continuity_seconds(generation_job, index, ranges), available_frames,
                _segment_visible_frames(start, end)) if segment_result is None else None)
            context_mode = (_segment_context_mode(generation_job.timeline, index, ranges)
                if frame_plan is not None else "off")
            context_frames = frame_plan.locked_frames if frame_plan is not None else 0
            generation_frames = frame_plan.generation_frames if frame_plan is not None else round(segment_duration * FPS)
            visible_frames = frame_plan.requested_frames if frame_plan is not None else _segment_visible_frames(start, end)
            visible_duration = visible_frames / FPS
            checkpoint_latent = None
            conditioning = None
            if segment_result is None:
                conditioning_inputs = {"clip": clip, "video_vae": video_vae, "audio_vae": audio_vae,
                    "generation_job": generation_job, "segment_index": index}
                if previous_images is not None:
                    conditioning_inputs["previous_images"] = previous_images
                    conditioning_inputs["previous_audio"] = previous_audio
                    if previous_latent is not None:
                        conditioning_inputs["previous_latent"] = previous_latent
                conditioning = stage_node("MiniMaxH3SegmentConditioning", f"{stage}_conditioning", **conditioning_inputs)
            if cache_mode == "复用已完成片段" and _cache_path(cache_files[index]).is_file():
                checkpoint = stage_node("MiniMaxH3SegmentCheckpointLoad", f"{stage}_cache_load",
                    cache_file=cache_files[index], preview_files=preview_files)
                if _latent_path(cache_files[index]).is_file():
                    checkpoint_latent = checkpoint.out(1)
            elif segment_result is not None:
                width, height = _video_size(generation_job.megapixels, generation_job.aspect_ratio)
                prepared = stage_node("MiniMaxH3SegmentResultPrepare", f"{stage}_result_prepare",
                    video=segment_result[0], duration_seconds=segment_duration, width=width, height=height)
                segment_video = stage_node("CreateVideo", f"{stage}_segment_video", images=prepared.out(0),
                    fps=float(FPS), audio=prepared.out(1))
                checkpoint = stage_node("MiniMaxH3SegmentCheckpoint", f"{stage}_checkpoint",
                    video=segment_video.out(0), cache_file=cache_files[index], preview_files=preview_files)
            else:
                noise = stage_node("RandomNoise", f"{stage}_noise",
                    noise_seed=(generation_job.seed + index) & 0xffffffffffffffff)
                guider = stage_node("BasicGuider", f"{stage}_guider", model=model, conditioning=conditioning.out(0))
                sigmas = stage_node("BasicScheduler", f"{stage}_scheduler", model=model,
                    scheduler=generation_job.scheduler, steps=generation_job.steps, denoise=generation_job.denoise)
                sampled = stage_node("MiniMaxH3SegmentSampler", f"{stage}_sampling", noise=noise.out(0),
                    guider=guider.out(0), sampler=sampler, sigmas=sigmas.out(0), latent_image=conditioning.out(1))
                decoded_images = stage_node("VAEDecode", f"{stage}_video_decode",
                    samples=sampled.out(0), vae=video_vae).out(0)
                decoded_audio = stage_node("VAEDecodeAudio", f"{stage}_audio_decode",
                    samples=sampled.out(0), vae=audio_vae).out(0)
                segment_video = stage_node("CreateVideo", f"{stage}_segment_video", images=decoded_images,
                    fps=float(FPS), audio=decoded_audio)
                checkpoint = stage_node("MiniMaxH3SegmentCheckpoint", f"{stage}_checkpoint",
                    video=segment_video.out(0), latent=sampled.out(0), cache_file=cache_files[index],
                    preview_files=preview_files)
                checkpoint_latent = sampled.out(0)
            entry_inputs = {"video": checkpoint.out(0), "segment_index": index,
                "start_time": start, "end_time": end,
                "context_frames": context_frames, "generation_frames": generation_frames,
                "visible_frames": visible_frames, "visible_duration": visible_duration,
                "cache_file": cache_files[index], "context_mode": context_mode}
            if conditioning is not None:
                entry_inputs["conditioning"] = conditioning.out(0)
            if checkpoint_latent is not None:
                entry_inputs["latent"] = checkpoint_latent
            entry = stage_node("MiniMaxH3SecondPassEntryPack", f"{stage}_second_pass_entry", **entry_inputs)
            batch_inputs = {"entry": entry.out(0)}
            if second_pass_batch is not None:
                batch_inputs["batch"] = second_pass_batch
            second_pass_batch = stage_node("MiniMaxH3SecondPassBatchAppend",
                f"{stage}_second_pass_batch", **batch_inputs).out(0)
            components = stage_node("GetVideoComponents", f"{stage}_components", video=checkpoint.out(0))
            images = components.out(0)
            audio = components.out(1)
            if segment_result is None:
                trimmed = stage_node("MiniMaxH3SegmentTrim", f"{stage}_trim", images=images,
                    audio=audio, context_frames=context_frames, duration_seconds=visible_duration,
                    generated_frames=generation_frames, visible_frames=visible_frames)
                images = trimmed.out(0)
                audio = trimmed.out(1)
            previous_images = images
            previous_audio = audio
            previous_latent = checkpoint_latent
                
            if accumulated_images is None:
                accumulated_images = images
                accumulated_audio = audio
            else:
                joined = stage_node(
                    "MiniMaxH3SegmentJoin", 
                    f"{stage}_join", 
                    previous_images=accumulated_images,
                    previous_audio=accumulated_audio, 
                    current_images=images, 
                    current_audio=audio
                )
                accumulated_images = joined.out(0)
                accumulated_audio = joined.out(1)
                
        video = stage_node("CreateVideo", f"segment_{len(ranges)}_of_{len(ranges)}_final_video",
                           images=accumulated_images, fps=float(FPS), audio=accumulated_audio)
        return io.NodeOutput(video.out(0), second_pass_batch, expand=graph.finalize())
