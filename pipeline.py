import json

import comfy.model_management
import comfy.sample
import comfy.utils
import torch

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .schema import ASPECT_RATIOS, CATEGORY, FPS, H3_GENERATION_JOB
from .segments import _context_frame_count, _segment_ranges, _segment_result
from .checkpoints import _cache_path, segment_cache_files
from .utils import _video_length, _video_size


SECOND_PASS_UPSCALE_METHODS = {
    "Lanczos（高质量）": "lanczos",
    "双三次": "bicubic",
    "双线性": "bilinear",
    "区域平均": "area",
    "最近邻": "nearest-exact",
}
SECOND_PASS_CROP_MODES = {"拉伸到目标比例": "disabled", "居中裁切到目标比例": "center"}


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
            io.Int.Input("generated_frames", min=0, default=0, optional=True)],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")])

    @classmethod
    def execute(cls, images, audio, context_frames, duration_seconds, generated_frames=0):
        if generated_frames and images.shape[0] != generated_frames:
            raise ValueError(f"模型解码得到 {images.shape[0]} 帧，但本次参考视频与生成目标均为 {generated_frames} 帧")
        context_frames = min(max(0, int(context_frames)), max(0, images.shape[0] - 1))
        images = images[context_frames:]
        frame_count = min(images.shape[0], max(1, round(duration_seconds * FPS)))
        sample_rate = audio["sample_rate"]
        audio_offset = min(audio["waveform"].shape[-1], round((context_frames / FPS) * sample_rate))
        waveform = audio["waveform"][..., audio_offset:]
        sample_count = min(waveform.shape[-1], round(duration_seconds * sample_rate))
        return io.NodeOutput(images[:frame_count], {**audio, "waveform": waveform[..., :sample_count]})


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
            
        images = torch.cat((previous_images, current_images), dim=0)

        waveform = torch.cat((previous_audio["waveform"], current_audio["waveform"]), dim=-1)
        return io.NodeOutput(images, {**current_audio, "waveform": waveform})


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
            outputs=[io.Video.Output(display_name="video")], 
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
        
        for index in range(len(ranges)):
            stage = f"segment_{index + 1}_of_{len(ranges)}"
            preview_files = json.dumps(cache_files[:index + 1])
            start, end = ranges[index]
            segment_duration = end - start
            segment_result = _segment_result(generation_job.timeline, start, end)
            available_frames = round((ranges[index - 1][1] - ranges[index - 1][0]) * FPS) if index else 0
            context_frames = _context_frame_count(generation_job.continuity_seconds, available_frames,
                round(segment_duration * FPS)) if segment_result is None else 0
            generation_frames = (_video_length(segment_duration + context_frames / FPS)
                if segment_result is None else round(segment_duration * FPS))
            if cache_mode == "复用已完成片段" and _cache_path(cache_files[index]).is_file():
                checkpoint = stage_node("MiniMaxH3SegmentCheckpointLoad", f"{stage}_cache_load",
                    cache_file=cache_files[index], preview_files=preview_files)
            elif segment_result is not None:
                width, height = _video_size(generation_job.megapixels, generation_job.aspect_ratio)
                prepared = stage_node("MiniMaxH3SegmentResultPrepare", f"{stage}_result_prepare",
                    video=segment_result[0], duration_seconds=segment_duration, width=width, height=height)
                segment_video = stage_node("CreateVideo", f"{stage}_segment_video", images=prepared.out(0),
                    fps=float(FPS), audio=prepared.out(1))
                checkpoint = stage_node("MiniMaxH3SegmentCheckpoint", f"{stage}_checkpoint",
                    video=segment_video.out(0), cache_file=cache_files[index], preview_files=preview_files)
            else:
                conditioning_inputs = {"clip": clip, "video_vae": video_vae, "audio_vae": audio_vae,
                    "generation_job": generation_job, "segment_index": index}
                if previous_images is not None:
                    conditioning_inputs["previous_images"] = previous_images
                    conditioning_inputs["previous_audio"] = previous_audio
                conditioning = stage_node("MiniMaxH3SegmentConditioning", f"{stage}_conditioning", **conditioning_inputs)
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
                    video=segment_video.out(0), cache_file=cache_files[index], preview_files=preview_files)
            components = stage_node("GetVideoComponents", f"{stage}_components", video=checkpoint.out(0))
            images = components.out(0)
            audio = components.out(1)
            if segment_result is None:
                trimmed = stage_node("MiniMaxH3SegmentTrim", f"{stage}_trim", images=images,
                    audio=audio, context_frames=context_frames, duration_seconds=segment_duration,
                    generated_frames=generation_frames)
                images = trimmed.out(0)
                audio = trimmed.out(1)
            previous_images = images
            previous_audio = audio
                
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
        return io.NodeOutput(video.out(0), expand=graph.finalize())
