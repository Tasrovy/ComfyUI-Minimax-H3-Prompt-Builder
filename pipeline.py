import json

import comfy.utils
import torch

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .schema import CATEGORY, FPS, H3_GENERATION_JOB
from .segments import _segment_ranges, _segment_result
from .checkpoints import _cache_path, segment_cache_files
from .utils import _video_size


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
            io.Int.Input("context_frames", min=0), io.Float.Input("duration_seconds", min=0.01, step=0.01)],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")])

    @classmethod
    def execute(cls, images, audio, context_frames, duration_seconds):
        context_frames = min(max(0, int(context_frames)), max(0, images.shape[0] - 1))
        images = images[context_frames:]
        frame_count = min(images.shape[0], max(1, round(duration_seconds * FPS)))
        sample_rate = audio["sample_rate"]
        audio_offset = min(audio["waveform"].shape[-1], round((context_frames / FPS) * sample_rate))
        waveform = audio["waveform"][..., audio_offset:]
        sample_count = min(waveform.shape[-1], round(duration_seconds * sample_rate))
        return io.NodeOutput(images[:frame_count], {**audio, "waveform": waveform[..., :sample_count]})


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
            description="自动逐段生成、实时预览、立即保存并拼接；停止后保留已完成片段，再次运行可从缓存继续。",
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
                    tooltip="更换模型、CLIP、VAE 或采样器设置后递增此值，以免复用旧模型生成的片段。"),
                io.Int.Input("preview_every_steps", display_name="实时预览间隔（步）", default=1, min=1, max=20)
            ],
            outputs=[io.Video.Output(display_name="video")], 
            enable_expand=True
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, sampler, generation_job,
                cache_mode="复用已完成片段", cache_version=0, preview_every_steps=1):
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
            segment_result = _segment_result(generation_job.timeline, *ranges[index])
            if cache_mode == "复用已完成片段" and _cache_path(cache_files[index]).is_file():
                checkpoint = stage_node("MiniMaxH3SegmentCheckpointLoad", f"{stage}_cache_load",
                    cache_file=cache_files[index], preview_files=preview_files)
            elif segment_result is not None:
                duration = ranges[index][1] - ranges[index][0]
                width, height = _video_size(generation_job.megapixels, generation_job.aspect_ratio)
                prepared = stage_node("MiniMaxH3SegmentResultPrepare", f"{stage}_result_prepare",
                    video=segment_result[0], duration_seconds=duration, width=width, height=height)
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
                sampled = stage_node("MiniMaxH3PreviewSampler", f"{stage}_sampling", noise=noise.out(0),
                    guider=guider.out(0), sampler=sampler, sigmas=sigmas.out(0), latent_image=conditioning.out(1),
                    preview_every_steps=preview_every_steps)
                decoded_images = stage_node("VAEDecode", f"{stage}_video_decode",
                    samples=sampled.out(0), vae=video_vae).out(0)
                decoded_audio = stage_node("VAEDecodeAudio", f"{stage}_audio_decode",
                    samples=sampled.out(0), vae=audio_vae).out(0)
                segment_duration = ranges[index][1] - ranges[index][0]
                trimmed = stage_node("MiniMaxH3SegmentTrim", f"{stage}_trim", images=decoded_images,
                    audio=decoded_audio, context_frames=conditioning.out(2), duration_seconds=segment_duration)
                segment_video = stage_node("CreateVideo", f"{stage}_segment_video", images=trimmed.out(0),
                    fps=float(FPS), audio=trimmed.out(1))
                checkpoint = stage_node("MiniMaxH3SegmentCheckpoint", f"{stage}_checkpoint",
                    video=segment_video.out(0), cache_file=cache_files[index], preview_files=preview_files)
            components = stage_node("GetVideoComponents", f"{stage}_components", video=checkpoint.out(0))
            images = components.out(0)
            audio = components.out(1)
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
