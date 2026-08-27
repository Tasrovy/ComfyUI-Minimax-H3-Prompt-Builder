from dataclasses import dataclass, replace

import torch
import comfy.samplers
import comfy.nested_tensor
import torchaudio
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (MiniMaxH3ReferenceToVideo as NativeRef2VA,
    _resize as resize_h3_image, align_frame_count, video_latent_t)

from .schema import (ASPECT_RATIOS, CATEGORY, EMPTY_SECTION_MODES, EMPTY_SECTION_OPTIONS, FPS,
    H3_GENERATION_JOB, H3_TIMELINE, GenerationJobData, MotionReferenceData, TimelineData, TrackListData)
from .timeline import MiniMaxH3FinalPrompt, _reference_subject_layout, _validate_timeline
from .utils import _match_reference_video, _sentence, _text, _time, _video_length


def _segment_drivers(timeline):
    actions = [clip for track in timeline.tracks.tracks if track.owner_kind == "actor" for clip in track.clips]
    body_actions = [clip for clip in actions if clip.kind == "body"]
    return body_actions or actions


def _segment_ranges(timeline):
    drivers = _segment_drivers(timeline)
    boundaries = {0.0, timeline.duration}
    for clip in drivers:
        if 1e-6 < clip.start_time < timeline.duration - 1e-6:
            boundaries.add(clip.start_time)
        if 1e-6 < clip.end_time < timeline.duration - 1e-6:
            boundaries.add(clip.end_time)
    points = sorted(boundaries)
    return tuple((points[index], points[index + 1]) for index in range(len(points) - 1))


def _segment_visible_frames(start, end):
    return max(1, round(end * FPS) - round(start * FPS))


def _segment_continuity_seconds(generation_job, segment_index, ranges=None):
    return generation_job.continuity_seconds if _segment_context_mode(
        generation_job.timeline, segment_index, ranges) != "off" else 0.0


def _segment_context_mode(timeline, segment_index, ranges=None):
    if segment_index <= 0:
        return "off"
    ranges = ranges or _segment_ranges(timeline)
    start = ranges[segment_index][0]
    starting = [clip for clip in _segment_drivers(timeline)
                if abs(clip.start_time - start) <= 1e-6]
    if not starting:
        return "full"
    choices = {"audio" if clip.audio_only_context else
        ("full" if clip.use_previous_context else "off") for clip in starting}
    if len(choices) > 1:
        raise ValueError(f"{start:g} 秒开始的动作片段的段间引导模式不一致")
    return choices.pop()


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


def _retime_timeline(timeline, duration):
    if abs(timeline.duration - duration) <= 1e-6:
        return timeline
    scale = duration / timeline.duration
    tracks = tuple(replace(track, clips=tuple(replace(clip,
        start_time=clip.start_time * scale, end_time=clip.end_time * scale) for clip in track.clips))
        for track in timeline.tracks.tracks)
    return replace(timeline, tracks=TrackListData(tracks), duration=duration)


@dataclass(frozen=True, slots=True)
class SegmentFramePlan:
    requested_frames: int
    current_frames: int
    locked_frames: int
    generation_frames: int

    @property
    def trailing_frames(self):
        return self.generation_frames - self.locked_frames - self.current_frames


_LATENT_CONTEXT_WINDOWS = (5, 22, 39, 56)


def _segment_frame_plan(seconds, available_frames, current_frames):
    requested_frames = max(1, int(current_frames))
    available_frames = max(0, int(available_frames))
    wanted = min(max(0, round(seconds * FPS)), available_frames)
    candidates = [value for value in _LATENT_CONTEXT_WINDOWS if value <= wanted]
    locked_frames = candidates[-1] if candidates else 0
    current_frames = _video_length(requested_frames / FPS)
    generation_frames = _video_length((current_frames + locked_frames) / FPS)
    return SegmentFramePlan(requested_frames, current_frames, locked_frames, generation_frames)


def _video_steps_for_frames(frame_count):
    if align_frame_count(frame_count) != frame_count:
        raise ValueError(f"{frame_count} 帧不能表示为完整的 MiniMax H3 视频 Latent 步")
    return video_latent_t(frame_count)


def _encode_context_audio(audio_vae, audio):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    return audio_vae.encode(waveform[:1].movedim(1, -1))


def _lock_context_prefix(latent, previous_images, previous_audio, previous_latent, video_vae, audio_vae,
                         locked_frames, width, height, lock_video=True):
    video, audio = (stream.clone() for stream in latent["samples"].unbind())
    video_steps = _video_steps_for_frames(locked_frames)
    previous_video = None
    previous_audio_latent = None
    if previous_latent is not None:
        previous_video, previous_audio_latent = previous_latent["samples"].unbind()
    if lock_video:
        if previous_images is not None and previous_images.shape[0]:
            tail = previous_images[-locked_frames:, ..., :3]
            if tail.shape[1] != height or tail.shape[2] != width:
                tail = resize_h3_image(tail, width, height, "disabled")
            tail_latent = video_vae.encode(tail)
            if tail_latent.shape[2] != video_steps:
                raise ValueError(f"{locked_frames} 帧段间画面编码得到 {tail_latent.shape[2]} 个 Latent 步，预期 {video_steps} 个")
        elif previous_video is not None:
            if previous_video.shape[1] != video.shape[1]:
                raise ValueError("上一片段视频 Latent 通道数与当前模型不一致")
            if previous_video.shape[3:] != video.shape[3:]:
                raise ValueError("上一片段与当前片段分辨率不一致，无法直接传递 Latent")
            if previous_video.shape[2] < video_steps:
                raise ValueError("上一片段视频 Latent 短于要求的段间锁定窗口")
            tail_latent = previous_video[:, :, -video_steps:]
        else:
            raise ValueError("段间画面锁定缺少上一片段画面或 Latent")
    if video_steps >= video.shape[2]:
        raise ValueError("段间锁定窗口必须短于本次生成视频")
    video_mask = torch.ones_like(video)
    if lock_video:
        video[:, :, :video_steps] = tail_latent.to(device=video.device, dtype=video.dtype)
        video_mask[:, :, :video_steps] = 0

    audio_mask = torch.ones_like(audio)
    audio_steps = max(1, round((locked_frames / FPS) * 40))
    if previous_audio is not None:
        sample_rate = int(previous_audio["sample_rate"])
        audio_samples = max(1, round((locked_frames / FPS) * sample_rate))
        tail_audio = {**previous_audio, "waveform": previous_audio["waveform"][..., -audio_samples:]}
        audio_latent = _encode_context_audio(audio_vae, tail_audio)
    elif previous_audio_latent is not None:
        if previous_audio_latent.shape[-1] < audio_steps:
            raise ValueError("上一片段音频 Latent 短于要求的段间锁定窗口")
        audio_latent = previous_audio_latent[..., -audio_steps:]
    else:
        audio_latent = None
    if audio_latent is not None:
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
    chinese = timeline.prompt_language == "中文"
    labels = {id(actor): f"{{{actor.actor_id}}}" for actor in timeline.characters.actors}
    states = []
    for track in timeline.tracks.tracks:
        if track.owner_kind == "audio":
            continue
        completed = [clip for clip in track.clips if clip.end_time <= start + 1e-6 and clip.result]
        if not completed:
            continue
        clip = max(completed, key=lambda item: item.end_time)
        owner = labels.get(id(track.owner), ({"environment": "环境", "camera": "镜头", "lighting": "灯光"}.get(track.owner_kind, "场景") if chinese else
            {"environment": "The environment", "camera": "The camera", "lighting": "The lighting"}.get(track.owner_kind, "The scene")))
        states.append(_sentence((f"本片段开始时，{owner}保持此前建立的状态：{clip.result}" if chinese else
            f"At the beginning of this segment, {owner} remains in this established state: {clip.result}")))
    return " ".join(states)


def _reference_bindings(timeline):
    bindings = []
    for track in timeline.tracks.tracks:
        for clip in track.clips:
            if track.owner_kind == "actor" and clip.motion_reference is not None:
                bindings.append(("actor", track, clip, clip.motion_reference.source))
            elif track.owner_kind == "camera" and clip.camera_reference is not None:
                bindings.append(("camera", track, clip, clip.camera_reference.source))
            elif track.owner_kind == "lighting" and clip.lighting_reference is not None:
                bindings.append(("lighting", track, clip, clip.lighting_reference.source))
            elif track.owner_kind == "environment" and clip.environment_reference is not None:
                bindings.append(("environment", track, clip, clip.environment_reference.source))
            elif track.owner_kind == "audio" and clip.audio_reference is not None:
                bindings.append(("audio", track, clip, clip.audio_reference.source))
    return bindings


def _reference_groups(timeline):
    groups = []
    by_key = {}
    for binding in _reference_bindings(timeline):
        kind, _, clip, source = binding
        key = (id(source), round(clip.start_time, 6), round(clip.end_time, 6))
        group = by_key.get(key)
        if group is None:
            group = {"source": source, "start": clip.start_time, "end": clip.end_time, "bindings": []}
            by_key[key] = group
            groups.append(group)
        if kind == "audio" and any(existing[0] == "audio" for existing in group["bindings"]):
            raise ValueError("同一参考视频在同一时间范围内只能绑定一个音频参考片段")
        group["bindings"].append(binding)
    return groups


def _fit_reference_group(group):
    source = group["source"]
    duration = group["end"] - group["start"]
    motion_frames = max(5, round(duration * FPS))
    aligned_frames = _video_length(duration)
    if aligned_frames > round(15.0 * FPS):
        raise ValueError("单个参考视频轨道片段不能超过 15 秒，请拆分片段")
    positions = torch.linspace(0, source.frames.shape[0] - 1, motion_frames,
        device=source.frames.device).round().long()
    frames = source.frames.index_select(0, positions)
    if aligned_frames > motion_frames:
        frames = torch.cat((frames, frames[-1:].repeat(aligned_frames - motion_frames, 1, 1, 1)), dim=0)

    has_audio = any(binding[0] == "audio" for binding in group["bindings"])
    audio = source.audio if has_audio else None
    if audio is not None:
        motion_samples = max(1, round(duration * audio["sample_rate"]))
        aligned_samples = round((aligned_frames / FPS) * audio["sample_rate"])
        waveform = audio["waveform"]
        shape = waveform.shape
        waveform = torch.nn.functional.interpolate(waveform.reshape(-1, 1, shape[-1]),
            size=motion_samples, mode="linear", align_corners=False).reshape(*shape[:-1], motion_samples)
        waveform = torch.nn.functional.pad(waveform, (0, max(0, aligned_samples - motion_samples)))
        audio = {**audio, "waveform": waveform[..., :aligned_samples]}

    roles = tuple(dict.fromkeys(binding[0] for binding in group["bindings"] if binding[0] != "audio"))
    owners = {id(binding[1].owner) for binding in group["bindings"] if binding[0] == "actor"}
    return MotionReferenceData(frames, audio, " + ".join(roles), source.source_duration,
        aligned_frames / FPS, duration, source=source, owner_id=next(iter(owners)) if len(owners) == 1 else None,
        clip_start=group["start"], clip_end=group["end"])


def _reference_media(timeline):
    video_groups = []
    audio_groups = []
    for group in _reference_groups(timeline):
        has_visual = any(binding[0] != "audio" for binding in group["bindings"])
        if has_visual:
            video_groups.append((group, _fit_reference_group(group)))
        else:
            source = group["source"]
            if source.audio is None:
                raise ValueError("音频参考所连接的参考视频不包含音频")
            audio_groups.append(group)
    return video_groups, audio_groups


def _reference_active_frames(reference):
    count = round((reference.motion_duration or reference.frames.shape[0] / FPS) * FPS)
    return reference.frames[:max(1, min(reference.frames.shape[0], count))]


def _reference_visible_frames(reference, duration):
    active = _reference_active_frames(reference)
    frame_count = max(1, round(duration * FPS))
    start = min(frame_count, max(0, round(reference.clip_start * FPS)))
    active = active[:max(0, frame_count - start)]
    if not active.shape[0]:
        return reference.frames[:1].repeat(frame_count, 1, 1, 1)
    rows = []
    if start:
        rows.append(active[:1].repeat(start, 1, 1, 1))
    if active.shape[0]:
        rows.append(active)
    used = start + active.shape[0]
    if used < frame_count:
        rows.append(active[-1:].repeat(frame_count - used, 1, 1, 1))
    return torch.cat(rows, dim=0)


def _tail_align_frames(frames, frame_count):
    if frames.shape[0] >= frame_count:
        return frames[-frame_count:]
    return torch.cat((frames[:1].repeat(frame_count - frames.shape[0], 1, 1, 1), frames), dim=0)


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


def _reference_visible_audio(reference, duration, sample_rate, channels, device, dtype):
    waveform = reference.audio["waveform"][:1].to(device=device, dtype=dtype)
    source_rate = int(reference.audio["sample_rate"])
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = _audio_channels(waveform, channels)
    active_samples = max(1, round(reference.motion_duration * sample_rate))
    waveform = waveform[..., :active_samples]
    total = max(1, round(duration * sample_rate))
    start = min(total, max(0, round(reference.clip_start * sample_rate)))
    waveform = waveform[..., :max(0, total - start)]
    return torch.nn.functional.pad(waveform, (start, max(0, total - start - waveform.shape[-1])))


def _align_motion_context(timeline, previous_timeline, references, frame_plan,
                          previous_images=None, previous_audio=None, context_mode="full"):
    if not references:
        return references
    target_frames = frame_plan.generation_frames
    if target_frames > round(15.0 * FPS):
        raise ValueError("参考视频加入段间上下文后超过 15 秒，请缩短片段或段间引导长度")

    previous = {}
    if previous_timeline is not None:
        for _, reference in _reference_media(previous_timeline)[0]:
            if reference.owner_id is not None and (reference.owner_id not in previous or
                    reference.clip_end >= previous[reference.owner_id].clip_end):
                previous[reference.owner_id] = reference

    aligned = []
    for reference in references:
        visible = (_reference_visible_frames(reference, timeline.duration) if frame_plan.locked_frames
            else _tail_align_frames(reference.frames, target_frames))
        locked = None
        previous_reference = previous.get(reference.owner_id)
        if context_mode == "full" and frame_plan.locked_frames and previous_images is not None and previous_images.shape[0]:
            locked = previous_images[-frame_plan.locked_frames:, ..., :3]
        elif context_mode == "full" and frame_plan.locked_frames and previous_reference is not None:
            previous_visible = _reference_visible_frames(previous_reference, previous_timeline.duration)
            locked = previous_visible[-frame_plan.locked_frames:]
        if frame_plan.locked_frames and locked is None:
            locked = visible[:1].repeat(frame_plan.locked_frames, 1, 1, 1)
        if locked is not None and locked.shape[1:3] != visible.shape[1:3]:
            locked = resize_h3_image(locked, visible.shape[2], visible.shape[1], "disabled")
        if locked is not None:
            locked = locked.to(device=visible.device, dtype=visible.dtype)
            if locked.shape[0] < frame_plan.locked_frames:
                locked = torch.cat((locked[:1].repeat(frame_plan.locked_frames - locked.shape[0], 1, 1, 1), locked), dim=0)
            locked = locked[-frame_plan.locked_frames:]
        if locked is not None:
            visible = torch.cat((locked, visible), dim=0)
        if visible.shape[0] < target_frames:
            visible = torch.cat((visible, visible[-1:].repeat(target_frames - visible.shape[0], 1, 1, 1)), dim=0)
        if visible.shape[0] != target_frames:
            raise ValueError(f"参考视频对齐后得到 {visible.shape[0]} 帧，本次模型生成需要 {target_frames} 帧")

        audio = reference.audio
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            waveform = audio["waveform"]
            channels = waveform.shape[1]
            if frame_plan.locked_frames:
                body = _reference_visible_audio(reference, timeline.duration, sample_rate, channels,
                    waveform.device, waveform.dtype)
            else:
                target_samples = round((target_frames / FPS) * sample_rate)
                body = waveform[..., -target_samples:]
                body = torch.nn.functional.pad(body, (target_samples - body.shape[-1], 0))
            locked_audio = None
            if frame_plan.locked_frames:
                locked_samples = round((frame_plan.locked_frames / FPS) * sample_rate)
                source_audio = previous_reference.audio if previous_reference is not None else None
                if source_audio is not None:
                    active_samples = round(previous_reference.motion_duration * source_audio["sample_rate"])
                    source_audio = {**source_audio, "waveform": source_audio["waveform"][..., :active_samples]}
                if source_audio is None:
                    source_audio = previous_audio
                locked_audio = _audio_tail(source_audio, sample_rate, locked_samples, channels,
                    waveform.device, waveform.dtype)
                body = torch.cat((locked_audio, body), dim=-1)
            target_samples = round((target_frames / FPS) * sample_rate)
            body = torch.nn.functional.pad(body, (0, max(0, target_samples - body.shape[-1])))[..., :target_samples]
            audio = {**audio, "waveform": body}

        aligned.append(replace(reference, frames=visible, audio=audio,
            aligned_duration=target_frames / FPS, context_duration=frame_plan.locked_frames / FPS,
            locked_duration=frame_plan.locked_frames / FPS if context_mode == "full" else 0.0))
    return aligned


def _validate_motion_alignment(references, target_frames):
    for index, reference in enumerate(references, 1):
        if reference.frames.shape[0] != target_frames:
            raise ValueError(f"<Video {index}> 有 {reference.frames.shape[0]} 帧，但本次模型生成需要 {target_frames} 帧")
        if reference.audio is None:
            continue
        sample_rate = int(reference.audio["sample_rate"])
        expected_samples = round((target_frames / FPS) * sample_rate)
        actual_samples = reference.audio["waveform"].shape[-1]
        if actual_samples != expected_samples:
            raise ValueError(f"<Video {index}> 的音频有 {actual_samples} 个采样点，但等长视频需要 {expected_samples} 个")


def _reference_subject_count(timeline):
    return _reference_subject_layout(timeline)[3]


def _opening_alignment_instruction(locked_duration, chinese=False, context_mode="full"):
    if locked_duration <= 1e-6:
        return ""
    if context_mode == "audio":
        return ((f"开头{_time(locked_duration)}秒的音频硬锁定为上一生成片段末尾的声音，画面不继承上一段。"
            f"在{_time(locked_duration)}秒处，当前片段音频从锁定声音连续衔接。") if chinese else
            (f"The opening {_time(locked_duration)} seconds of audio are hard-locked to the preceding generated segment's ending sound; "
            f"the visuals do not inherit the preceding segment. At {_time(locked_duration)} seconds, the current audio continues seamlessly from the locked sound."))
    if chinese:
        return (f"开头{_time(locked_duration)}秒硬锁定为上一生成片段末尾的动作、姿态与取景。"
            f"在{_time(locked_duration)}秒处，当前动作从锁定的最后一帧直接延续，并一直进行到本片段最后一帧。")
    return (f"The opening {_time(locked_duration)} seconds are hard-locked to the preceding generated segment's final motion, pose, and framing. "
        f"At {_time(locked_duration)} seconds, the current action continues directly from the locked final frame and runs through the final frame")


def _semantic_references(timeline, video_groups, audio_groups, first_subject_number,
                         prefix_duration=0.0, has_locked_context=False, chinese=False):
    actor_subjects = _reference_subject_layout(timeline)[0]
    labels = {id(actor): f"{{{actor.actor_id}}}" for actor in timeline.characters.actors}
    instructions = []
    definitions = []
    retentions = []
    action_labels = []
    camera_labels = []
    lighting_labels = []
    environment_labels = []
    expression_labels = []
    gaze_labels = []
    audio_labels = []
    subject_number = first_subject_number
    audio_number = 1
    audio_numbers = {}
    for video_number, (group, reference) in enumerate(video_groups, 1):
        if reference.audio is not None:
            audio_numbers[id(group)] = audio_number
            audio_number += 1
    for group in audio_groups:
        audio_numbers[id(group)] = audio_number
        audio_number += 1

    for video_number, (group, _) in enumerate(video_groups, 1):
        bindings = group["bindings"]
        has_camera = any(binding[0] == "camera" for binding in bindings)
        has_environment = any(binding[0] == "environment" for binding in bindings)
        scene_locked = has_environment
        if prefix_duration or has_camera:
            roles = []
            if prefix_duration:
                roles.append(("动作过渡上下文与当前镜头对齐的时间顺序" if has_locked_context else "当前镜头对齐的时间顺序") if chinese else
                    ("motion-transition context and current shot-aligned temporal order" if has_locked_context
                        else "current shot-aligned temporal order"))
            if has_camera:
                roles.append("镜头运动与取景变化" if chinese else "camera movement and framing progression")
            if has_environment:
                roles.append("环境外观与空间布局" if chinese else "environment appearance and spatial layout")
            role_text = "以及".join(roles) if chinese else " and ".join(roles)
            definitions.append((f"<Video {video_number}>是[Shot 1]的{role_text}参考。" if chinese else
                f"<Video {video_number}> is the reference for [Shot 1]'s {role_text}."))
            if chinese:
                retention = (f"<Video {video_number}>（[Shot 1]的时间结构）：fully_preserved - 保留其{role_text}，"
                    + ("只替换已声明的源人物身份与服装。" if scene_locked else "但不导入源视频中的人物身份、服装或环境。"))
            else:
                retention = (f"<Video {video_number}> (temporal structure in [Shot 1]): fully_preserved - Preserve its {role_text} "
                    + ("and replace only the declared source performer." if scene_locked else
                        "without importing source identities, clothing, or environment."))
            retentions.append(retention)
        for kind, track, clip, _ in bindings:
            if kind == "actor":
                owner = labels.get(id(track.owner), "该人物" if chinese else "the character")
                target = (f"<Subject {actor_subjects[id(track.owner)]}> ({owner})"
                    if id(track.owner) in actor_subjects else owner)
                subject = f"<Subject {subject_number}>"
                person_id = _text(clip.motion_reference.person_id)
                person_description = _text(clip.motion_reference.person_description)
                performance_name = {"body": ("肢体表演" if chinese else "body performance"),
                    "expression": ("面部表演" if chinese else "facial performance"),
                    "gaze": ("视线表演" if chinese else "gaze performance")}.get(clip.kind,
                        "表演" if chinese else "performance")
                if person_description:
                    source_person = ((f"源人物{person_id}（{person_description}）" if person_id else person_description) if chinese else
                        (f"source performer {person_id} ({person_description})" if person_id else person_description))
                    definitions.append((f"{subject}是从<Video {video_number}>中的{source_person}提取并迁移给{target}的{performance_name}。" if chinese else
                        f"{subject} is the {performance_name} of {source_person} derived from <Video {video_number}> and transferred to {target}."))
                else:
                    definitions.append((f"{subject}是从<Video {video_number}>提取并迁移给{target}的{performance_name}。" if chinese else
                        f"{subject} is the {performance_name} derived from <Video {video_number}> and transferred to {target}."))
                if clip.kind == "expression":
                    retention = (f"保留面部表情顺序、变化时机与最终表情，迁移给{target}。" if chinese else
                        f"Transfer its expression order, timing, and final expression to {target}.")
                    instruction = ((f"使用{subject}作为{target}的权威面部表演参考。完整保留表情变化顺序、时机与最终表情，"
                        f"同时保留{owner}已声明的身份与固定外观。" + ("只替换源人物。" if scene_locked else "不复制源视频中的人物身份、服装、背景或灯光。"))
                        if chinese else (f"Use {subject} as {target}'s authoritative facial performance reference. Preserve its expression order, timing, "
                        f"and final expression while retaining {owner}'s declared identity and fixed appearance. " +
                        ("Replace only the source performer." if scene_locked else "Do not copy source identity, clothing, background, or lighting.")))
                    expression_labels.append(subject)
                elif clip.kind == "gaze":
                    retention = (f"保留视线方向、转移时机与最终视线，迁移给{target}。" if chinese else
                        f"Transfer its gaze direction, timing, and final gaze to {target}.")
                    instruction = ((f"使用{subject}作为{target}的权威视线表演参考。完整保留视线方向与时机，同时保留{owner}的身份。" if chinese else
                        f"Use {subject} as {target}'s authoritative gaze-performance reference. Preserve its gaze direction and timing while retaining {owner}'s identity.") )
                    gaze_labels.append(subject)
                else:
                    retention = (f"将动作顺序、时序与最终姿态迁移给{target}。" if chinese else
                        f"Transfer its motion order, timing, and final pose to {target}.")
                    instruction = ((f"使用{subject}作为{target}的权威肢体表演参考。完整保留其动作顺序、时序、重心转移与最终姿态，"
                        f"同时保留{owner}已声明的身份与固定外观。" + ("只替换源人物。" if scene_locked else "只迁移肢体表演，不复制源视频中的人物、面部、服装、背景、灯光或音频。"))
                        if chinese else (f"Use {subject} as {target}'s authoritative body-performance reference. Preserve its complete motion order, timing, weight shifts, "
                        f"and final pose while retaining {owner}'s declared identity and fixed appearance. " +
                        ("Replace only the source performer." if scene_locked else "Transfer only body performance; do not copy source performer, face, clothing, background, lighting, or audio.")))
                retentions.append((f"{subject}（出现在[Shot 1]）：attribute_transfer - {retention}" if chinese else
                    f"{subject} (appears in [Shot 1]): attribute_transfer - {retention}"))
                line = instruction
                if person_description:
                    line += ((f" 在<Video {video_number}>中只跟随{source_person}的表演，不要混用其他可见人物的动作。") if chinese else
                        (f" In <Video {video_number}>, follow only {source_person}'s performance and do not mix motion from other visible performers."))
                instructions.append((id(clip), _sentence(line)))
                if clip.kind == "body": action_labels.append(subject)
                subject_number += 1
            elif kind == "camera":
                line = ((f"使用<Video {video_number}>作为权威的镜头运动、取景与时间结构参考。"
                    + ("只替换源人物，保留源视频的环境与空间布局。" if scene_locked else "只迁移镜头行为，不复制源视频中的表演者、服装、背景、灯光或音频。")) if chinese else
                    (f"Use <Video {video_number}> as the authoritative camera movement, framing, and temporal-structure reference. "
                    + ("Replace only the source performer while preserving the source environment and spatial layout." if scene_locked else
                        "Transfer only camera behavior; do not copy the source performer, clothing, background, lighting, or audio.")))
                instructions.append((id(clip), _sentence(line)))
                camera_labels.append(f"<Video {video_number}>")
            elif kind == "lighting":
                subject = f"<Subject {subject_number}>"
                definitions.append((f"{subject}是从<Video {video_number}>提取并迁移到[Shot 1]的灯光行为。" if chinese else
                    f"{subject} is the lighting behavior derived from <Video {video_number}> and transferred to [Shot 1]."))
                retentions.append(((f"{subject}（出现在[Shot 1]）：fully_preserved - 保留源视频的光线方向、颜色、强度与变化节奏，并与源环境保持一致。"
                    if chinese else f"{subject} (appears in [Shot 1]): fully_preserved - Preserve the source light direction, color, intensity, and change rhythm with the source environment.") if scene_locked else
                    (f"{subject}（出现在[Shot 1]）：attribute_transfer - 迁移其光线方向、颜色、强度与变化节奏，但不导入源场景。" if chinese else
                    f"{subject} (appears in [Shot 1]): attribute_transfer - Transfer its light direction, color, intensity, and change rhythm without importing the source scene.")))
                line = (f"使用{subject}作为[Shot 1]的权威灯光行为参考，同时保留源视频环境并只替换源人物。" if chinese and scene_locked else
                    f"Use {subject} as [Shot 1]'s authoritative lighting behavior while preserving the source environment and replacing only the source performer." if scene_locked else
                    (f"使用{subject}作为[Shot 1]的权威灯光行为参考，同时保留已声明的主体与环境。" if chinese else
                        f"Use {subject} as [Shot 1]'s authoritative lighting behavior while preserving the declared subjects and environment."))
                instructions.append((id(clip), _sentence(line)))
                lighting_labels.append(subject)
                subject_number += 1
            elif kind == "environment":
                subject = f"<Subject {subject_number}>"
                definitions.append((f"{subject}是从<Video {video_number}>提取的源视频环境与空间布局。" if chinese else
                    f"{subject} is the source video's environment and spatial layout derived from <Video {video_number}>."))
                retentions.append((f"{subject}（出现在[Shot 1]）：fully_preserved - 保留源视频环境、空间布局与背景关系，只替换源人物。" if chinese else
                    f"{subject} (appears in [Shot 1]): fully_preserved - Preserve the source environment, spatial layout, and background relationship while replacing only the source performer."))
                instructions.append((id(clip), _sentence((f"使用{subject}作为[Shot 1]的权威环境与空间布局参考，保留源视频背景，只替换源人物。" if chinese else
                    f"Use {subject} as [Shot 1]'s authoritative environment and spatial-layout reference. Preserve the source background and replace only the source performer."))))
                environment_labels.append(subject)
                subject_number += 1
            elif kind == "audio":
                number = audio_numbers[id(group)]
                label = f"<Audio {number}>"
                definitions.append((f"{label}是与<Video {video_number}>关联音轨中的{clip.audio_type}声音参考。" if chinese else
                    f"{label} is the {clip.audio_type} sound reference from the soundtrack associated with <Video {video_number}>."))
                retentions.append((f"{label}：reference - 遵循其时序与声音特征，但不复制源信号。" if chinese else
                    f"{label}: reference - Follow its timing and sound character without copying the source signal."))
                instructions.append((id(clip), _sentence((f"使用{label}作为{clip.audio_type}的时序与声音特征参考。" if chinese else
                    f"Use {label} as the {clip.audio_type} timing and sound-character reference."))))
                audio_labels.append(label)

    standalone_audios = []
    for group in audio_groups:
        binding = group["bindings"][0]
        _, _, clip, source = binding
        number = audio_numbers[id(group)]
        label = f"<Audio {number}>"
        definitions.append((f"{label}是所提供参考视频音轨中的{clip.audio_type}声音参考。" if chinese else
            f"{label} is the {clip.audio_type} sound reference from the supplied reference video soundtrack."))
        retentions.append((f"{label}：reference - 遵循其时序与声音特征，但不复制源信号。" if chinese else
            f"{label}: reference - Follow its timing and sound character without copying the source signal."))
        instructions.append((id(clip), _sentence((f"使用{label}作为{clip.audio_type}的时序与声音特征参考。" if chinese else
            f"Use {label} as the {clip.audio_type} timing and sound-character reference."))))
        audio_labels.append(label)
        standalone_audios.append(source.audio)

    instruction_map = {}
    for clip_id, instruction in instructions:
        instruction_map[clip_id] = " ".join(filter(_text, (instruction_map.get(clip_id, ""), instruction)))
    summary_parts = []
    if action_labels:
        summary_parts.append((f"肢体表演迁移自{'、'.join(action_labels)}。" if chinese else
            f"Body performance is transferred from {' and '.join(action_labels)}."))
    if expression_labels:
        summary_parts.append((f"面部表演迁移自{'、'.join(expression_labels)}。" if chinese else
            f"Facial performance is transferred from {' and '.join(expression_labels)}."))
    if gaze_labels:
        summary_parts.append((f"视线表演迁移自{'、'.join(gaze_labels)}。" if chinese else
            f"Gaze performance is transferred from {' and '.join(gaze_labels)}."))
    if environment_labels:
        summary_parts.append((f"环境与空间布局保留自{'、'.join(environment_labels)}。" if chinese else
            f"Environment and spatial layout are preserved from {' and '.join(environment_labels)}."))
    if camera_labels:
        labels_text = "、".join(dict.fromkeys(camera_labels)) if chinese else " and ".join(dict.fromkeys(camera_labels))
        summary_parts.append((f"镜头运动与时间结构遵循{labels_text}。" if chinese else
            f"Camera movement and temporal structure follow {labels_text}."))
    if lighting_labels:
        summary_parts.append((f"灯光行为迁移自{'、'.join(lighting_labels)}。" if chinese else
            f"Lighting behavior is transferred from {' and '.join(lighting_labels)}."))
    if audio_labels:
        labels_text = "、".join(dict.fromkeys(audio_labels)) if chinese else " and ".join(dict.fromkeys(audio_labels))
        summary_parts.append((f"声音参考自{labels_text}。" if chinese else f"Sound is referenced from {labels_text}."))
    return instruction_map, "\n".join(definitions), "\n".join(retentions), " ".join(summary_parts), standalone_audios


def _compile_generation_segment(generation_job, segment_index, frame_plan=None,
                                previous_images=None, previous_audio=None):
    ranges = _segment_ranges(generation_job.timeline)
    if segment_index < 0 or segment_index >= len(ranges):
        raise ValueError(f"分段编号超出范围：{segment_index}")
    start, end = ranges[segment_index]
    context_mode = _segment_context_mode(generation_job.timeline, segment_index, ranges)
    if frame_plan is None:
        available = round((ranges[segment_index - 1][1] - ranges[segment_index - 1][0]) * FPS) if segment_index else 0
        frame_plan = _segment_frame_plan(_segment_continuity_seconds(generation_job, segment_index, ranges),
            available, round((end - start) * FPS))
    timeline = _retime_timeline(_segment_timeline(generation_job.timeline, start, end),
        frame_plan.current_frames / FPS)
    previous_timeline = (_segment_timeline(generation_job.timeline, *ranges[segment_index - 1])
        if segment_index else None)
    state = (_persistent_state(generation_job.timeline, start)
        if context_mode == "full" and frame_plan.locked_frames > 0 else "")
    prefix_duration = frame_plan.locked_frames / FPS
    locked_duration = frame_plan.locked_frames / FPS
    video_groups, audio_groups = _reference_media(timeline)
    media_references = [reference for _, reference in video_groups]
    motion_instructions, motion_definitions, motion_retentions, motion_summary, standalone_audios = _semantic_references(
        timeline, video_groups, audio_groups, _reference_subject_count(timeline) + 1, prefix_duration,
        context_mode == "full" and frame_plan.locked_frames > 0, timeline.prompt_language == "中文")
    if len(media_references) > 3:
        raise ValueError("单个生成片段最多支持 3 段语义参考视频")
    if sum(reference.audio is not None for reference in media_references) + len(standalone_audios) > 3:
        raise ValueError("单个生成片段最多支持 3 段参考音频")
        
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
        suppress_initial_state=context_mode == "full" and frame_plan.locked_frames > 0,
        generation_duration=frame_plan.generation_frames / FPS,
        timeline_offset=prefix_duration,
        opening_instructions=_opening_alignment_instruction(locked_duration, timeline.prompt_language == "中文", context_mode),
        continuation=context_mode == "full" and frame_plan.locked_frames > 0,
    )[0]
    if compiled.video_settings.length != frame_plan.generation_frames:
        raise ValueError(f"提示词生成长度为 {compiled.video_settings.length} 帧，分段计划需要 {frame_plan.generation_frames} 帧")
    media_references = _align_motion_context(timeline, previous_timeline, media_references, frame_plan,
        previous_images, previous_audio, context_mode)
    _validate_motion_alignment(media_references, compiled.video_settings.length)
    return compiled, media_references, standalone_audios


def _empty_sections_mode(value):
    if value in EMPTY_SECTION_MODES:
        return value
    if value in (1, "1", "不输出"):
        return "自动补全"
    if value in (2, "2"):
        return "输出 N/A"
    return "自动补全"


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
            io.Combo.Input("empty_sections", display_name="声音空节处理", options=EMPTY_SECTION_OPTIONS, default="自动补全"),
            io.Float.Input("continuity_seconds", display_name="段间引导长度（秒）", default=0.92, min=0.21, max=2.33, step=0.01)],
            outputs=[H3_GENERATION_JOB.Output(display_name="generation_job")])

    @classmethod
    def execute(cls, timeline, megapixels, aspect_ratio, seed, scheduler, steps, denoise, ref_image_size,
                continuity_seconds, empty_sections="自动补全"):
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
            io.Image.Input("previous_images", optional=True), io.Audio.Input("previous_audio", optional=True),
            io.Latent.Input("previous_latent", optional=True)],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(),
                io.Int.Output(display_name="context_frames")])

    @classmethod
    def execute(cls, clip, video_vae, audio_vae, generation_job, segment_index, previous_images=None,
                previous_audio=None, previous_latent=None):
        start, end = _segment_ranges(generation_job.timeline)[segment_index]
        frame_plan = _segment_frame_plan(_segment_continuity_seconds(generation_job, segment_index),
            previous_images.shape[0] if previous_images is not None else 0, _segment_visible_frames(start, end))
        compiled, motion_references, standalone_audios = _compile_generation_segment(generation_job, segment_index, frame_plan,
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
            ref_videos=ref_videos, ref_video_audios=ref_video_audios,
            ref_audios={f"ref_audio_{index}": audio for index, audio in enumerate(standalone_audios)})
        positive, latent = native[0], native[1]
        if frame_plan.locked_frames:
            latent = _lock_context_prefix(latent, previous_images, previous_audio, previous_latent,
                video_vae, audio_vae, frame_plan.locked_frames, settings.width, settings.height,
                lock_video=_segment_context_mode(generation_job.timeline, segment_index) == "full")
        return io.NodeOutput(positive, latent, frame_plan.locked_frames)


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
            description="按多段 Ref2VA 生成顺序预览提示词，并输出带标注的参考图。",
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
            available = _segment_visible_frames(*ranges[index - 1]) if index else 0
            frame_plan = _segment_frame_plan(_segment_continuity_seconds(generation_job, index, ranges), available,
                _segment_visible_frames(start, end))
            compiled, motion_references, standalone_audios = _compile_generation_segment(generation_job, index, frame_plan)
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
                prefix_parts = []
                if reference.locked_duration:
                    prefix_parts.append(f"锁定上下文 {_time(reference.locked_duration)} 秒")
                prefix = ("＋".join(prefix_parts) + "＋") if prefix_parts else ""
                references.append(f"<Video {1 + offset}> = 当前片段语义参考视频 {offset + 1} [{reference.role}]（{prefix}"
                    f"当前引用 {_time(reference.motion_duration or reference.frames.shape[0] / FPS)} 秒，"
                    f"送入模型 {_time(reference.aligned_duration or reference.frames.shape[0] / FPS)} 秒）")
            paired_audio_count = sum(reference.audio is not None for reference in motion_references)
            for offset, _ in enumerate(standalone_audios, paired_audio_count + 1):
                references.append(f"<Audio {offset}> = 当前片段独立音频语义参考")
            if frame_plan.locked_frames:
                context_mode = _segment_context_mode(generation_job.timeline, index, ranges)
                trailing = (f"并裁掉尾部 {frame_plan.trailing_frames} 帧对齐冗余" if frame_plan.trailing_frames else "")
                references.insert(0, (f"段间引导 = 仅强锁定上一片段末尾对应的 {_time(frame_plan.locked_frames / FPS)} 秒音频，"
                    f"画面不继承上一段；生成后裁掉前 {frame_plan.locked_frames} 帧对应时长{trailing}（不占用参考视频槽位）"
                    if context_mode == "audio" else
                    f"段间引导 = Latent 强锁定上一片段末尾 {frame_plan.locked_frames} 帧及对应音频，"
                    f"生成后裁掉前 {frame_plan.locked_frames} 帧{trailing}（不占用参考视频槽位）"))
            
            header = [f"========== 片段 {index + 1}/{len(ranges)} ==========",
                f"时间轴范围：{_time(start)}–{_time(end)} 秒 | 裁剪后输出：{frame_plan.requested_frames} 帧 / {_time(frame_plan.requested_frames / FPS)} 秒"]
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
