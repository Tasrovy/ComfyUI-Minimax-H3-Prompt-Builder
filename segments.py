from dataclasses import replace

import torch
import comfy.samplers
import comfy.nested_tensor
import torchaudio
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (MiniMaxH3ReferenceToVideo as NativeRef2VA,
    _resize as resize_h3_image)

from .schema import (ASPECT_RATIOS, CATEGORY, EMPTY_SECTION_MODES, EMPTY_SECTION_OPTIONS, FPS,
    H3_GENERATION_JOB, H3_TIMELINE, GenerationJobData, TimelineData, TrackListData)
from .timeline import MiniMaxH3FinalPrompt, _validate_timeline
from .utils import _match_reference_video, _same_image, _sentence, _text, _time


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


def _segment_result(timeline, start, end):
    results = [clip for track in timeline.tracks.tracks if track.owner_kind == "actor" for clip in track.clips
        if clip.rendered_video is not None and clip.end_time > start + 1e-6 and clip.start_time < end - 1e-6]
    if len(results) > 1:
        raise ValueError(f"时间轴 {start:g}–{end:g} 秒的同一生成片段只能绑定一个已生成结果")
    if not results:
        return None
    clip = results[0]
    return clip.rendered_video, clip.rendered_video_version


def _segment_timeline(timeline, start, end):
    tracks = []
    for track in timeline.tracks.tracks:
        clips = []
        for clip in track.clips:
            if clip.end_time <= start + 1e-6 or clip.start_time >= end - 1e-6:
                continue
            clips.append(replace(clip, start_time=max(clip.start_time, start) - start,
                end_time=min(clip.end_time, end) - start))
        if clips:
            tracks.append(replace(track, clips=tuple(clips)))
    return replace(timeline, tracks=TrackListData(tuple(tracks)), duration=end - start)


def _context_frame_count(seconds, available_frames):
    wanted = min(max(0, round(seconds * 24)), max(0, int(available_frames)))
    if wanted < 5:
        return 0
    return 5 + 17 * ((wanted - 5) // 17)


def _encode_context_audio(audio_vae, audio):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    return audio_vae.encode(waveform[:1].movedim(1, -1))


def _lock_context_prefix(latent, previous_images, previous_audio, video_vae, audio_vae,
                         context_frames, width, height):
    video, audio = (stream.clone() for stream in latent["samples"].unbind())
    tail = previous_images[-context_frames:, ..., :3]
    if tail.shape[1] != height or tail.shape[2] != width:
        tail = resize_h3_image(tail, width, height, "disabled")
    tail_latent = video_vae.encode(tail).to(device=video.device, dtype=video.dtype)
    video_steps = min(tail_latent.shape[2], video.shape[2] - 1)
    video[:, :, :video_steps] = tail_latent[:, :, :video_steps]
    video_mask = torch.ones_like(video)
    video_mask[:, :, :video_steps] = 0
    for offset, weight in enumerate((0.55, 0.34, 0.16)):
        step = video_steps + offset
        if step >= video.shape[2]:
            break
        video[:, :, step] = video[:, :, step] * (1.0 - weight) + video[:, :, step - 1] * weight

    audio_mask = torch.ones_like(audio)
    if previous_audio is not None:
        sample_rate = int(previous_audio["sample_rate"])
        audio_samples = max(1, round((context_frames / 24.0) * sample_rate))
        tail_audio = {**previous_audio, "waveform": previous_audio["waveform"][..., -audio_samples:]}
        audio_latent = _encode_context_audio(audio_vae, tail_audio)
        audio_latent = audio_latent.to(device=audio.device, dtype=audio.dtype)
        audio_steps = min(audio_latent.shape[-1], audio.shape[-1] - 1)
        audio[..., :audio_steps] = audio_latent[..., :audio_steps]
        audio_mask[..., :audio_steps] = 0

    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return out


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


def _motion_reference_bindings(timeline):
    return [(id(track.owner), clip) for track in timeline.tracks.tracks if track.owner_kind == "actor"
        for clip in track.clips if clip.motion_reference is not None and _text(clip.content)]


def _reference_active_frames(reference):
    count = round((reference.motion_duration or reference.frames.shape[0] / FPS) * FPS)
    return reference.frames[:max(1, min(reference.frames.shape[0], count))]


def _reference_visible_frames(clip, duration):
    active = _reference_active_frames(clip.motion_reference)
    frame_count = max(1, round(duration * FPS))
    start = min(frame_count, max(0, round(clip.start_time * FPS)))
    active = active[:max(0, frame_count - start)]
    if not active.shape[0]:
        return clip.motion_reference.frames[:1].repeat(frame_count, 1, 1, 1)
    rows = []
    if start:
        rows.append(active[:1].repeat(start, 1, 1, 1))
    if active.shape[0]:
        rows.append(active)
    used = start + active.shape[0]
    if used < frame_count:
        rows.append(active[-1:].repeat(frame_count - used, 1, 1, 1))
    return torch.cat(rows, dim=0)


def _audio_channels(waveform, channels):
    if waveform.shape[1] == channels:
        return waveform
    if waveform.shape[1] == 1:
        return waveform.repeat(1, channels, 1)
    return waveform[:, :channels]


def _audio_tail(audio, sample_rate, samples, channels, device, dtype):
    if audio is None or audio.get("waveform") is None:
        return torch.zeros((1, channels, samples), device=device, dtype=dtype)
    waveform = audio["waveform"][:1].to(device=device, dtype=dtype)
    source_rate = int(audio["sample_rate"])
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = _audio_channels(waveform, channels)
    waveform = waveform[..., -samples:]
    return torch.nn.functional.pad(waveform, (samples - waveform.shape[-1], 0))


def _reference_visible_audio(clip, duration, sample_rate, channels, device, dtype):
    reference = clip.motion_reference
    waveform = reference.audio["waveform"][:1].to(device=device, dtype=dtype)
    source_rate = int(reference.audio["sample_rate"])
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = _audio_channels(waveform, channels)
    active_samples = max(1, round((reference.motion_duration or clip.end_time - clip.start_time) * sample_rate))
    waveform = waveform[..., :active_samples]
    total = max(1, round(duration * sample_rate))
    start = min(total, max(0, round(clip.start_time * sample_rate)))
    waveform = waveform[..., :max(0, total - start)]
    return torch.nn.functional.pad(waveform, (start, max(0, total - start - waveform.shape[-1])))


def _align_motion_context(timeline, previous_timeline, references, context_frames, target_frames,
                          previous_images=None, previous_audio=None):
    if not references:
        return references
    if target_frames > round(15.0 * FPS):
        raise ValueError("动作参考视频加入段间上下文后超过 15 秒，请缩短片段或段间引导长度")

    previous = {}
    for owner, clip in _motion_reference_bindings(previous_timeline) if previous_timeline is not None else ():
        if owner not in previous or clip.end_time >= previous[owner].end_time:
            previous[owner] = clip

    aligned = []
    for (owner, clip), reference in zip(_motion_reference_bindings(timeline), references):
        visible = _reference_visible_frames(clip, timeline.duration)
        prefix = None
        previous_clip = previous.get(owner)
        if context_frames and previous_clip is not None:
            previous_visible = _reference_visible_frames(previous_clip, previous_timeline.duration)
            prefix = previous_visible[-context_frames:]
        elif context_frames and previous_images is not None and previous_images.shape[0]:
            prefix = previous_images[-context_frames:, ..., :3]
        if context_frames:
            if prefix is None or not prefix.shape[0]:
                prefix = visible[:1].repeat(context_frames, 1, 1, 1)
            if prefix.shape[1:3] != visible.shape[1:3]:
                prefix = resize_h3_image(prefix, visible.shape[2], visible.shape[1], "disabled")
            prefix = prefix.to(device=visible.device, dtype=visible.dtype)
            if prefix.shape[0] < context_frames:
                prefix = torch.cat((prefix[:1].repeat(context_frames - prefix.shape[0], 1, 1, 1), prefix), dim=0)
            visible = torch.cat((prefix[-context_frames:], visible), dim=0)
        if visible.shape[0] > target_frames:
            visible = visible[:target_frames]
        elif visible.shape[0] < target_frames:
            visible = torch.cat((visible, visible[-1:].repeat(target_frames - visible.shape[0], 1, 1, 1)), dim=0)

        audio = reference.audio
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            waveform = audio["waveform"]
            channels = waveform.shape[1]
            body = _reference_visible_audio(clip, timeline.duration, sample_rate, channels,
                waveform.device, waveform.dtype)
            if context_frames:
                prefix_samples = round((context_frames / FPS) * sample_rate)
                source_audio = previous_clip.motion_reference.audio if previous_clip is not None else None
                if source_audio is None:
                    source_audio = previous_audio
                prefix_audio = _audio_tail(source_audio, sample_rate, prefix_samples, channels,
                    waveform.device, waveform.dtype)
                body = torch.cat((prefix_audio, body), dim=-1)
            target_samples = round((target_frames / FPS) * sample_rate)
            body = torch.nn.functional.pad(body[..., :target_samples],
                (0, max(0, target_samples - body.shape[-1])))
            audio = {**audio, "waveform": body}

        aligned.append(replace(reference, frames=visible, audio=audio,
            aligned_duration=target_frames / FPS, context_duration=context_frames / FPS))
    return aligned


def _reference_subject_count(timeline):
    references = [actor.card.reference for actor in timeline.characters.actors]
    references.extend((timeline.style.reference, timeline.environment.card.reference))
    unique = []
    for reference in filter(None, references):
        if not any(_same_image(reference.image, item.image) for item in unique):
            unique.append(reference)
    return len(unique)


def _motion_references(timeline, first_video_number, first_subject_number, has_context=False):
    role_text = {
        "仅动作": ("body motion", False),
        "动作与镜头": ("body motion", True),
        "完整表演": ("complete performance", True),
        "动作与声音": ("performance", False),
    }
    labels = {id(actor): f"{actor.card.name} (S{index})" for index, actor in enumerate(timeline.characters.actors, 1)}
    references = []
    instructions = []
    definitions = []
    retentions = []
    summary_labels = []
    camera_labels = []
    video_number = first_video_number
    audio_number = 1
    subject_number = first_subject_number
    for track in timeline.tracks.tracks:
        if track.owner_kind != "actor":
            continue
        for clip in track.clips:
            reference = clip.motion_reference
            if reference is None or not _text(clip.content):
                continue
            owner = labels.get(id(track.owner), "the character")
            responsibility, uses_camera = role_text[reference.role]
            if has_context:
                subject_definition = (f"<Subject {subject_number}> is {owner}'s motion sequence derived from <Video {video_number}>, "
                    f"beginning with continuity and followed by the current {responsibility}.")
                line = (f"Transfer <Subject {subject_number}> to {owner}: continue the preceding motion, then reproduce the current "
                    f"{responsibility} in the supplied order. Preserve {owner}'s declared identity, clothing, and scene.")
            else:
                subject_definition = f"<Subject {subject_number}> is {owner}'s {responsibility} derived from <Video {video_number}>."
                line = (f"Transfer <Subject {subject_number}> to {owner}. Preserve its complete order, "
                    f"timing, and final pose while rendering {owner}'s declared identity, clothing, and scene.")
            definitions.append(subject_definition)
            retentions.append(f"<Subject {subject_number}> (appears in [Shot 1]): attribute_transfer - "
                f"Transfer its aligned performance and timing to {owner}.")
            summary_labels.append(f"<Subject {subject_number}>")
            if uses_camera:
                definitions.append(f"<Video {video_number}> provides the camera behavior and temporal structure for [Shot 1].")
                retentions.append(f"<Video {video_number}> (camera and temporal structure in [Shot 1]): partially_preserved - "
                    "Preserve its camera behavior and temporal order.")
                line += f" Follow <Video {video_number}>'s camera behavior and temporal structure."
                camera_labels.append(f"<Video {video_number}>")
            if reference.audio is not None:
                line += f" Use its synchronized <Audio {audio_number}>."
                definitions.append(f"<Audio {audio_number}> is the synchronized sound reference paired with <Video {video_number}>.")
                retentions.append(f"<Audio {audio_number}>: reference - Use its synchronized timing and sound character.")
                audio_number += 1
            instructions.append(_sentence(line))
            references.append(reference)
            video_number += 1
            subject_number += 1
    summary = "" if not summary_labels else f"The action is guided by {' and '.join(summary_labels)}."
    if camera_labels:
        summary += f" The camera and temporal structure are guided by {' and '.join(camera_labels)}."
    return references, " ".join(instructions), "\n".join(definitions), "\n".join(retentions), summary


def _compile_generation_segment(generation_job, segment_index, context_frames=0,
                                previous_images=None, previous_audio=None):
    ranges = _segment_ranges(generation_job.timeline)
    if segment_index < 0 or segment_index >= len(ranges):
        raise ValueError(f"分段编号超出范围：{segment_index}")
    start, end = ranges[segment_index]
    
    timeline = _segment_timeline(generation_job.timeline, start, end)
    previous_timeline = (_segment_timeline(generation_job.timeline, *ranges[segment_index - 1])
        if segment_index else None)
    state = _persistent_state(generation_job.timeline, start)
    motion_references, motion_instructions, motion_definitions, motion_retentions, motion_summary = _motion_references(
        timeline, 1, _reference_subject_count(timeline) + 1, context_frames > 0)
    full_performance_actors = {id(track.owner) for track in timeline.tracks.tracks if track.owner_kind == "actor"
        for clip in track.clips if clip.motion_reference is not None and clip.motion_reference.role == "完整表演"}
    if len(motion_references) > 3:
        raise ValueError("单个生成片段最多支持 3 段动作参考视频")
        
    compiled = MiniMaxH3FinalPrompt.execute(
        timeline, 
        generation_job.megapixels, 
        generation_job.aspect_ratio,
        prompt_format="Ref", 
        additional_instructions=state,
        motion_instructions=motion_instructions,
        motion_definitions=motion_definitions,
        motion_retentions=motion_retentions,
        motion_summary=motion_summary,
        empty_sections=generation_job.empty_sections,
        continuity_keyframe=False,
        suppress_initial_state=context_frames > 0,
        suppress_actor_state_ids=full_performance_actors,
        generation_duration=timeline.duration + context_frames / FPS
    )[0]
    motion_references = _align_motion_context(timeline, previous_timeline, motion_references, context_frames,
        compiled.video_settings.length, previous_images, previous_audio)
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
            io.Float.Input("continuity_seconds", display_name="段间引导长度（秒）", default=0.92, min=0.21, max=2.33, step=0.01)],
            outputs=[H3_GENERATION_JOB.Output(display_name="generation_job")])

    @classmethod
    def execute(cls, timeline, megapixels, aspect_ratio, seed, scheduler, steps, denoise, ref_image_size,
                continuity_seconds, empty_sections="不输出"):
        if not isinstance(timeline, TimelineData):
            raise TypeError("生成任务包需要 MiniMax H3 时间轴")
        empty_sections = _empty_sections_mode(empty_sections)
        _validate_timeline(timeline)
        _segment_ranges(timeline)
        return io.NodeOutput(GenerationJobData(timeline, megapixels, aspect_ratio, seed, scheduler, steps, denoise,
            ref_image_size, continuity_seconds, empty_sections))


class MiniMaxH3SegmentConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SegmentConditioning", display_name="MiniMax H3 分段条件（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Clip.Input("clip"), io.Vae.Input("video_vae"),
            io.Vae.Input("audio_vae"), H3_GENERATION_JOB.Input("generation_job"), io.Int.Input("segment_index"),
            io.Image.Input("previous_images", optional=True), io.Audio.Input("previous_audio", optional=True)],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(),
                io.Int.Output(display_name="context_frames")])

    @classmethod
    def execute(cls, clip, video_vae, audio_vae, generation_job, segment_index, previous_images=None, previous_audio=None):
        context_frames = _context_frame_count(generation_job.continuity_seconds,
            previous_images.shape[0] if previous_images is not None else 0)
        compiled, motion_references = _compile_generation_segment(generation_job, segment_index, context_frames,
            previous_images, previous_audio)
        settings = compiled.video_settings
        ref_videos = {}
        ref_video_audios = {}
        ref_images = {f"ref_image_{index}": item.image for index, item in enumerate(compiled.references)}
        for video_index, reference in enumerate(motion_references):
            frames = (_match_reference_video(reference.frames, settings.width, settings.height)
                if generation_job.ref_image_size == "match" else reference.frames)
            ref_videos[f"ref_video_{video_index}"] = frames
            if reference.audio is not None:
                ref_video_audios[f"ref_video_audio_{video_index}"] = reference.audio
        native = NativeRef2VA.execute(clip=clip, vae=video_vae, audio_vae=audio_vae, prompt=compiled.text,
            width=settings.width, height=settings.height, length=settings.length,
            ref_image_size=generation_job.ref_image_size,
            ref_images=ref_images,
            ref_videos=ref_videos, ref_video_audios=ref_video_audios)
        positive, latent = native[0], native[1]
        if context_frames:
            latent = _lock_context_prefix(latent, previous_images, previous_audio, video_vae, audio_vae,
                context_frames, settings.width, settings.height)
        return io.NodeOutput(positive, latent, context_frames)


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
            available = round((ranges[index - 1][1] - ranges[index - 1][0]) * 24) if index else 0
            context_frames = _context_frame_count(generation_job.continuity_seconds, available)
            compiled, motion_references = _compile_generation_segment(generation_job, index, context_frames)
            rendered = _segment_result(generation_job.timeline, start, end)
            settings = compiled.video_settings
            
            # 1. 静态参考图输出
            for ref_item in compiled.references:
                if ref_item.image is not None and ref_item.picture_number not in seen_numbers:
                    seen_numbers.add(ref_item.picture_number)
                    label = f"<Picture {ref_item.picture_number}>: {ref_item.role}"
                    labeled_images.append(_draw_picture_label(ref_item.image, label))

            # 2. 媒体与段间引导清单
            references = []
            for ref_item in compiled.references:
                references.append(f"<Picture {ref_item.picture_number}> = [{ref_item.role}] ({ref_item.usage or 'Default'})")
            for offset, reference in enumerate(motion_references):
                context = reference.context_duration
                prefix = (f"上下文 {_time(context)} 秒（优先上一段同人物动作参考，否则上一段生成视频尾部）＋"
                    if context else "")
                references.append(f"<Video {1 + offset}> = 当前片段动作参考视频 {offset + 1}（{prefix}"
                    f"当前片段动作 {_time(reference.motion_duration or reference.frames.shape[0] / FPS)} 秒，"
                    f"送入模型 {_time(reference.aligned_duration or reference.frames.shape[0] / FPS)} 秒）")
            if context_frames:
                references.insert(0,
                    f"段间引导 = 硬锁定上一片段末尾 {context_frames} 帧及对应音频（不占用参考视频槽位）")
            
            header = [f"========== 片段 {index + 1}/{len(ranges)} ==========",
                f"时间轴范围：{_time(start)}–{_time(end)} 秒 | 输出时长：{_time(end - start)} 秒"]
            if rendered:
                header.append(f"生成方式：使用已生成结果（版本 {rendered[1]}），跳过模型采样")
            header.extend(["【媒体绑定清单】：",
                *(references or ["参考媒体：无"]),
                "",
                compiled.text
            ])
            sections.append("\n".join(header))

        preview_text = "\n\n".join(sections)
        return io.NodeOutput(preview_text, labeled_images, ui={"text": (preview_text,)})
