from dataclasses import replace

import comfy.samplers
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo as NativeRef2VA

from .schema import (ASPECT_RATIOS, CATEGORY, EMPTY_SECTION_MODES, EMPTY_SECTION_OPTIONS,
    H3_GENERATION_JOB, H3_TIMELINE, GenerationJobData, TimelineData, TrackListData)
from .timeline import MiniMaxH3FinalPrompt, _validate_timeline
from .utils import _match_reference_video, _sentence, _text, _time


def _segment_ranges(timeline):
    drivers = [clip for track in timeline.tracks.tracks if track.owner_kind == "actor" for clip in track.clips]
    if not drivers:
        drivers = [clip for track in timeline.tracks.tracks if track.owner_kind == "environment" for clip in track.clips]
    if len(drivers) < 2:
        return ((0.0, timeline.duration),)
    groups = []
    for clip in sorted(drivers, key=lambda item: (item.start_time, item.end_time)):
        if not groups or clip.start_time >= groups[-1][1] - 1e-6:
            groups.append([clip.start_time, clip.end_time])
        else:
            groups[-1][1] = max(groups[-1][1], clip.end_time)
    boundaries = [group[1] for group in groups[:-1]]
    boundaries = [value for index, value in enumerate(boundaries)
                  if value > (boundaries[index - 1] if index else 0.0) + 1e-6 and value < timeline.duration - 1e-6]
    points = [0.0, *boundaries, timeline.duration]
    return tuple((points[index], points[index + 1]) for index in range(len(points) - 1))


def _segment_timeline(timeline, start, end, leading_seconds=0.0):
    tracks = []
    for track in timeline.tracks.tracks:
        clips = []
        for clip in track.clips:
            if clip.end_time <= start + 1e-6 or clip.start_time >= end - 1e-6:
                continue
            clips.append(replace(clip, start_time=max(clip.start_time, start) - start + leading_seconds,
                end_time=min(clip.end_time, end) - start + leading_seconds))
        if clips:
            tracks.append(replace(track, clips=tuple(clips)))
    return replace(timeline, tracks=TrackListData(tuple(tracks)), duration=end - start + leading_seconds)


def _persistent_state(timeline, start):
    if start <= 1e-6:
        return ""
    labels = {id(actor): f"{actor.card.name} (S{index})" for index, actor in enumerate(timeline.characters.actors, 1)}
    states = []
    for track in timeline.tracks.tracks:
        completed = [clip for clip in track.clips if clip.end_time <= start + 1e-6 and clip.result]
        if not completed:
            continue
        clip = max(completed, key=lambda item: item.end_time)
        owner = labels.get(id(track.owner), "The environment" if track.owner_kind == "environment" else "The scene")
        states.append(_sentence(f"At the beginning of this segment, {owner} remains in this established state: {clip.result}"))
    return " ".join(states)


def _motion_references(timeline, first_video_number):
    role_text = {
        "仅动作": "Use it only for body mechanics, motion trajectory, timing, weight transfer, contact timing, and gesture rhythm. Do not copy identity, face, hair, clothing, background, lighting, or camera composition from it.",
        "动作与镜头": "Use it for body mechanics, motion timing, and camera movement. Do not copy identity, face, hair, clothing, background, or lighting from it.",
        "完整表演": "Use it for body mechanics, facial performance, interaction timing, and camera movement. Preserve the declared subjects' identities, clothing, and environment instead of copying those elements from it.",
        "动作与声音": "Use it for body mechanics, performance timing, and its paired reference audio. Do not copy identity, clothing, or background from it.",
    }
    labels = {id(actor): f"{actor.card.name} (S{index})" for index, actor in enumerate(timeline.characters.actors, 1)}
    references = []
    instructions = []
    video_number = first_video_number
    audio_number = 1
    for track in timeline.tracks.tracks:
        if track.owner_kind != "actor":
            continue
        for clip in track.clips:
            reference = clip.motion_reference
            if reference is None or not _text(clip.content):
                continue
            owner = labels.get(id(track.owner), "the character")
            line = (f"For {owner}'s action from {_time(clip.start_time)} to {_time(clip.end_time)} seconds, "
                    f"use <Video {video_number}> as the motion reference. {role_text[reference.role]} "
                    f"The performer in the reference video is replaced by {owner}: render {owner}'s identity, "
                    "face, hair, body, clothing, and environment from the declared references instead.")
            if reference.audio is not None:
                line += f" Its paired sound is <Audio {audio_number}>."
                audio_number += 1
            instructions.append(_sentence(line))
            references.append(reference)
            video_number += 1
    return references, " ".join(instructions)


def _compile_generation_segment(generation_job, segment_index, has_previous_segment):
    ranges = _segment_ranges(generation_job.timeline)
    if segment_index < 0 or segment_index >= len(ranges):
        raise ValueError(f"分段编号超出范围：{segment_index}")
    start, end = ranges[segment_index]
    
    timeline = _segment_timeline(generation_job.timeline, start, end, leading_seconds=0.0)
    state = _persistent_state(generation_job.timeline, start)
    first_video_number = 2 if has_previous_segment else 1
    motion_references, motion_instructions = _motion_references(timeline, first_video_number)
    
    continuity_instruction = ""
    if has_previous_segment:
        continuity_instruction = _sentence(
            "Seamlessly continue the motion, posture, lighting, and camera velocity from <Video 1>. "
            "Execute the segment's actions starting immediately from 0.00 seconds without pause or replay"
        )
    
    total_videos = len(motion_references) + (1 if has_previous_segment else 0)
    if total_videos > 3:
        raise ValueError("单个生成片段最多支持 3 段参考视频；后续片段需为连续性视频保留 1 个位置，因此最多连接 2 个动作参考视频")
        
    additional = " ".join(filter(_text, (state, continuity_instruction)))
    compiled = MiniMaxH3FinalPrompt.execute(
        timeline, 
        generation_job.megapixels, 
        generation_job.aspect_ratio,
        prompt_format="Ref", 
        additional_instructions=additional, 
        motion_instructions=motion_instructions,
        empty_sections=generation_job.empty_sections, 
        continuity_keyframe=has_previous_segment
    )[0]
    return compiled, motion_references


def _empty_sections_mode(value):
    if value in EMPTY_SECTION_MODES:
        return value
    if value in (1, "1"):
        return "不输出"
    if value in (2, "2"):
        return "输出 N/A"
    return "不输出"


class MiniMaxH3GenerationJob(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3GenerationJob", display_name="MiniMax H3 生成任务包（Generation Job）", category=CATEGORY, inputs=[
            H3_TIMELINE.Input("timeline"),
            io.Float.Input("megapixels", display_name="百万像素", default=0.98, min=0.01, max=16.0, step=0.01),
            io.Combo.Input("aspect_ratio", display_name="宽高比", options=list(ASPECT_RATIOS), default="16:9"),
            io.Int.Input("seed", display_name="噪声种子", default=0, min=0, max=0xffffffffffffffff,
                control_after_generate=True),
            io.Combo.Input("scheduler", display_name="调度器", options=comfy.samplers.SCHEDULER_NAMES, default="simple"),
            io.Int.Input("steps", display_name="采样步数", default=4, min=1, max=10000),
            io.Float.Input("denoise", display_name="降噪强度", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Combo.Input("ref_image_size", display_name="参考媒体尺寸", options=["match", "max"], default="match"),
            io.Combo.Input("empty_sections", display_name="空节处理", options=EMPTY_SECTION_OPTIONS, default="不输出"),
            io.Float.Input("continuity_seconds", display_name="连续性参考长度（秒）", default=2.0, min=0.25, max=15.0, step=0.25),
            io.Float.Input("overlap_seconds", display_name="重叠匹配长度（秒）", default=0.5, min=0.1, max=2.0, step=0.05)],
            outputs=[H3_GENERATION_JOB.Output(display_name="generation_job")])

    @classmethod
    def execute(cls, timeline, megapixels, aspect_ratio, seed, scheduler, steps, denoise, ref_image_size,
                continuity_seconds, overlap_seconds, empty_sections="不输出"):
        if not isinstance(timeline, TimelineData):
            raise TypeError("生成任务包需要 MiniMax H3 时间轴")
        empty_sections = _empty_sections_mode(empty_sections)
        _validate_timeline(timeline)
        _segment_ranges(timeline)
        return io.NodeOutput(GenerationJobData(timeline, megapixels, aspect_ratio, seed, scheduler, steps, denoise,
            ref_image_size, continuity_seconds, overlap_seconds, empty_sections))


class MiniMaxH3SegmentConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentConditioning", display_name="MiniMax H3 分段条件（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Clip.Input("clip"), io.Vae.Input("video_vae"),
            io.Vae.Input("audio_vae"), H3_GENERATION_JOB.Input("generation_job"), io.Int.Input("segment_index"),
            io.Image.Input("previous_tail_video", optional=True)],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()])

    @classmethod
    def execute(cls, clip, video_vae, audio_vae, generation_job, segment_index, previous_tail_video=None):
        compiled, motion_references = _compile_generation_segment(generation_job, segment_index,
            previous_tail_video is not None)
        settings = compiled.video_settings
        ref_videos = {}
        ref_video_audios = {}
        ref_images = {f"ref_image_{index}": item.image for index, item in enumerate(compiled.references)}
        video_index = 0
        if previous_tail_video is not None:
            tail_video = (_match_reference_video(previous_tail_video, settings.width, settings.height)
                if generation_job.ref_image_size == "match" else previous_tail_video)
            ref_videos["ref_video_0"] = tail_video
            ref_images[f"ref_image_{len(compiled.references)}"] = tail_video[-1:]
            video_index = 1
        for reference in motion_references:
            frames = (_match_reference_video(reference.frames, settings.width, settings.height)
                if generation_job.ref_image_size == "match" else reference.frames)
            ref_videos[f"ref_video_{video_index}"] = frames
            if reference.audio is not None:
                ref_video_audios[f"ref_video_audio_{video_index}"] = reference.audio
            video_index += 1
        return NativeRef2VA.execute(clip=clip, vae=video_vae, audio_vae=audio_vae, prompt=compiled.text,
            width=settings.width, height=settings.height, length=settings.length,
            ref_image_size=generation_job.ref_image_size,
            ref_images=ref_images,
            ref_videos=ref_videos, ref_video_audios=ref_video_audios)


def _draw_picture_label(tensor_image, label_text):
    pil_img = TF.to_pil_image(tensor_image.squeeze(0).permute(2, 0, 1).cpu().clamp(0, 1))
    draw = ImageDraw.Draw(pil_img)
    font_size = max(18, int(pil_img.height * 0.04))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        
    bbox = draw.textbbox((12, 12), label_text, font=font)
    draw.rectangle((bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6), fill=(0, 0, 0, 210), outline=(255, 215, 0), width=2)
    draw.text((12, 12), label_text, fill=(255, 255, 255), font=font)
    return TF.to_tensor(pil_img).permute(1, 2, 0).unsqueeze(0)


def _create_placeholder_image(width, height, title, subtitle):
    img = Image.new("RGB", (width, height), color=(25, 28, 36))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, width - 9, height - 9], outline=(70, 80, 100), width=2)
    draw.line([(0, 0), (width, height)], fill=(40, 45, 60), width=1)
    draw.line([(0, height), (width, 0)], fill=(40, 45, 60), width=1)
    
    font_title_size = max(22, int(height * 0.055))
    font_sub_size = max(14, int(height * 0.035))
    try:
        font_t = ImageFont.truetype("arial.ttf", font_title_size)
        font_s = ImageFont.truetype("arial.ttf", font_sub_size)
    except Exception:
        font_t = ImageFont.load_default()
        font_s = font_t
        
    t_box = draw.textbbox((0, 0), title, font=font_t)
    s_box = draw.textbbox((0, 0), subtitle, font=font_s)
    
    t_x, t_y = (width - (t_box[2] - t_box[0])) // 2, height // 2 - 30
    s_x, s_y = (width - (s_box[2] - s_box[0])) // 2, height // 2 + 15
    
    draw.rectangle([min(t_x, s_x) - 16, t_y - 12, max(t_x + t_box[2], s_x + s_box[2]) + 16, s_y + s_box[3] + 12],
                   fill=(15, 18, 22), outline=(0, 200, 255), width=2)
    draw.text((t_x, t_y), title, fill=(0, 220, 255), font=font_t)
    draw.text((s_x, s_y), subtitle, fill=(180, 190, 205), font=font_s)
    return TF.to_tensor(img).permute(1, 2, 0).unsqueeze(0)


class MiniMaxH3PromptPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptPreview", 
            display_name="MiniMax H3 最终提示词预览（Final Prompt Preview）",
            category=CATEGORY, 
            description="按多段 Ref2VA 生成顺序预览提示词，并输出带标注的静态图及动态帧占位图。",
            inputs=[H3_GENERATION_JOB.Input("generation_job")], 
            outputs=[
                io.String.Output(display_name="final_prompts"),
                io.Image.Output(display_name="labeled_reference_images", is_output_list=True)
            ],
            is_output_node=True
        )

    @classmethod
    def execute(cls, generation_job):
        if not isinstance(generation_job, GenerationJobData):
            raise TypeError("最终提示词预览需要 MiniMax H3 生成任务包")
            
        ranges = _segment_ranges(generation_job.timeline)
        sections = []
        labeled_images = []
        seen_numbers = set()

        for index, (start, end) in enumerate(ranges):
            has_previous = index > 0
            compiled, motion_references = _compile_generation_segment(generation_job, index, has_previous)
            settings = compiled.video_settings
            
            # 1. 静态参考图输出
            for ref_item in compiled.references:
                if ref_item.image is not None and ref_item.picture_number not in seen_numbers:
                    seen_numbers.add(ref_item.picture_number)
                    label = f"<Picture {ref_item.picture_number}>: {ref_item.role}"
                    labeled_images.append(_draw_picture_label(ref_item.image, label))

            # 2. 动态尾帧占位图
            continuity_pic_num = len(compiled.references) + 1
            if has_previous and continuity_pic_num not in seen_numbers:
                seen_numbers.add(continuity_pic_num)
                placeholder = _create_placeholder_image(
                    width=settings.width,
                    height=settings.height,
                    title=f"<Picture {continuity_pic_num}>: Dynamic Opening Frame",
                    subtitle="[Auto-captured from previous segment tail at runtime]"
                )
                labeled_images.append(placeholder)

            # 3. 文本清单
            references = []
            for ref_item in compiled.references:
                references.append(f"<Picture {ref_item.picture_number}> = [{ref_item.role}] ({ref_item.usage or 'Default'})")
            if has_previous:
                references.append("<Video 1> = 上一片段尾部连续性视频")
                references.append(f"<Picture {continuity_pic_num}> = 上一片段尾帧（运行时自动填入）")
                
            first_motion_number = 2 if has_previous else 1
            references.extend(
                f"<Video {first_motion_number + offset}> = 当前片段动作参考视频 {offset + 1}"
                for offset in range(len(motion_references))
            )
            
            header = [
                f"========== 片段 {index + 1}/{len(ranges)} ==========",
                f"时间轴范围：{_time(start)}–{_time(end)} 秒 | 生成时长：{_time(end - start)} 秒",
                "【媒体绑定清单】：",
                *(references or ["参考媒体：无"]),
                "",
                compiled.text
            ]
            sections.append("\n".join(header))

        preview_text = "\n\n".join(sections)
        return io.NodeOutput(preview_text, labeled_images, ui={"text": (preview_text,)})