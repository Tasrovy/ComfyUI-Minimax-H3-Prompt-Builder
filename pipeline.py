import torch

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .schema import CATEGORY, FPS, H3_GENERATION_JOB, GenerationJobData
from .segments import _segment_ranges


class MiniMaxH3ContinuityTail(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3ContinuityTail", display_name="MiniMax H3 连续性尾部视频（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Image.Input("images"),
            io.Float.Input("continuity_seconds", default=2.0, min=0.25, max=15.0, step=0.25)],
            outputs=[io.Image.Output(display_name="tail_video")])

    @classmethod
    def execute(cls, images, continuity_seconds):
        frame_count = min(images.shape[0], max(5, round(continuity_seconds * FPS)))
        tail = images[-frame_count:]
        if tail.shape[0] < 5:
            tail = torch.cat((tail, tail[-1:].repeat(5 - tail.shape[0], 1, 1, 1)), dim=0)
        return io.NodeOutput(tail)


class MiniMaxH3SegmentTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentTrim", display_name="MiniMax H3 片段裁切（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Image.Input("images"), io.Audio.Input("audio"),
            io.Float.Input("duration_seconds", min=0.01, step=0.01)],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")])

    @classmethod
    def execute(cls, images, audio, duration_seconds):
        frame_count = min(images.shape[0], max(1, round(duration_seconds * FPS)))
        sample_rate = audio["sample_rate"]
        sample_count = min(audio["waveform"].shape[-1], round(duration_seconds * sample_rate))
        return io.NodeOutput(images[:frame_count], {**audio, "waveform": audio["waveform"][..., :sample_count]})


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
                io.Audio.Input("current_audio"),
                io.Float.Input("overlap_seconds", default=0.5, min=0.1, max=2.0, step=0.05)
            ],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")]
        )

    @classmethod
    def execute(cls, previous_images, previous_audio, current_images, current_audio, overlap_seconds):
        if previous_images.shape[1:] != current_images.shape[1:]:
            raise ValueError("分段画面尺寸不一致，无法拼接")
        if previous_audio["sample_rate"] != current_audio["sample_rate"]:
            raise ValueError("分段音频采样率不一致，无法拼接")
            
        # 画面平滑拼接：消除首帧完全重叠，同时对接缝帧做 50% 均值混合消除单帧曝光/色温闪烁
        if previous_images.shape[0] >= 1 and current_images.shape[0] >= 2:
            seam_frame = 0.5 * previous_images[-1:] + 0.5 * current_images[:1]
            images = torch.cat((previous_images[:-1], seam_frame, current_images[1:]), dim=0)
        else:
            images = torch.cat((previous_images, current_images[1:]), dim=0)

        waveform = torch.cat((previous_audio["waveform"], current_audio["waveform"]), dim=-1)
        return io.NodeOutput(images, {**current_audio, "waveform": waveform})


class MiniMaxH3MultiSegmentGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiSegmentGenerate", 
            display_name="MiniMax H3 多段生成（Decoded Video）",
            category=CATEGORY, 
            description="自动逐段生成、解码并拼接；节点内保留每个已完成步骤的耗时，并显示当前步骤耗时。",
            inputs=[
                io.Model.Input("model"), 
                io.Clip.Input("clip"), 
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"), 
                io.Sampler.Input("sampler"), 
                H3_GENERATION_JOB.Input("generation_job")
            ],
            outputs=[io.Video.Output(display_name="video")], 
            enable_expand=True
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, sampler, generation_job):
        ranges = _segment_ranges(generation_job.timeline)
        graph = GraphBuilder()
        parent_node_id = cls.hidden.unique_id if cls.hidden is not None else None

        def stage_node(class_type, node_id, **inputs):
            node = graph.node(class_type, id=node_id, **inputs)
            if parent_node_id is not None:
                node.set_override_display_id(parent_node_id)
            return node

        accumulated_images = None
        accumulated_audio = None
        previous_tail_video = None
        
        for index in range(len(ranges)):
            stage = f"segment_{index + 1}_of_{len(ranges)}"
            conditioning_inputs = {
                "clip": clip, 
                "video_vae": video_vae, 
                "audio_vae": audio_vae,
                "generation_job": generation_job, 
                "segment_index": index
            }
            if previous_tail_video is not None:
                conditioning_inputs["previous_tail_video"] = previous_tail_video
                
            conditioning = stage_node("MiniMaxH3SegmentConditioning", f"{stage}_conditioning", **conditioning_inputs)
            noise = stage_node("RandomNoise", f"{stage}_noise", noise_seed=(generation_job.seed + index) & 0xffffffffffffffff)
            guider = stage_node("BasicGuider", f"{stage}_guider", model=model, conditioning=conditioning.out(0))
            sigmas = stage_node("BasicScheduler", f"{stage}_scheduler", model=model, scheduler=generation_job.scheduler,
                               steps=generation_job.steps, denoise=generation_job.denoise)
            sampled = stage_node("SamplerCustomAdvanced", f"{stage}_sampling", noise=noise.out(0),
                                 guider=guider.out(0), sampler=sampler, sigmas=sigmas.out(0), latent_image=conditioning.out(1))
            images = stage_node("VAEDecode", f"{stage}_video_decode", samples=sampled.out(0), vae=video_vae).out(0)
            audio = stage_node("VAEDecodeAudio", f"{stage}_audio_decode", samples=sampled.out(0), vae=audio_vae).out(0)
            
            # 真实时长控制：后续片段多裁出 1 帧冗余，用于在 SegmentJoin 中完美抵消首帧重叠
            segment_duration = ranges[index][1] - ranges[index][0]
            trim_duration = segment_duration + ((1.0 / FPS) if index > 0 else 0.0)
            
            trimmed = stage_node("MiniMaxH3SegmentTrim", f"{stage}_trim", images=images, audio=audio, duration_seconds=trim_duration)
            images = trimmed.out(0)
            audio = trimmed.out(1)
            
            if index < len(ranges) - 1:
                previous_tail_video = stage_node(
                    "MiniMaxH3ContinuityTail", 
                    f"{stage}_continuity", 
                    images=images,
                    continuity_seconds=generation_job.continuity_seconds
                ).out(0)
                
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
                    current_audio=audio,
                    overlap_seconds=generation_job.overlap_seconds
                )
                accumulated_images = joined.out(0)
                accumulated_audio = joined.out(1)
                
        video = stage_node("CreateVideo", f"segment_{len(ranges)}_of_{len(ranges)}_final_video",
                           images=accumulated_images, fps=float(FPS), audio=accumulated_audio)
        return io.NodeOutput(video.out(0), expand=graph.finalize())