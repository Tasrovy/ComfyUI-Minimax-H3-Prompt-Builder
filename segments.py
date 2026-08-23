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


_LATENT_CONTEXT_WINDOWS = (5, 22, 39, 56)


def _segment_frame_plan(seconds, available_frames, current_frames):
    requested_frames = max(1, int(current_frames))
    available_frames = max(0, int(available_frames))
    wanted = min(max(0, round(seconds * FPS)), available_frames)
    candidates = [value for value in _LATENT_CONTEXT_WINDOWS if value <= wanted]
    locked_frames = candidates[-1] if candidates else 0
    generation_frames = _video_length((requested_frames + locked_frames) / FPS)
    return SegmentFramePlan(requested_frames, generation_frames - locked_frames, locked_frames,
        generation_frames)


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
                         locked_frames, width, height):
    video, audio = (stream.clone() for stream in latent["samples"].unbind())
    video_steps = _video_steps_for_frames(locked_frames)
    if previous_latent is not None:
        previous_video, previous_audio_latent = previous_latent["samples"].unbind()
        if previous_video.shape[1] != video.shape[1]:
            raise ValueError("上一片段视频 Latent 通道数与当前模型不一致")
        if previous_video.shape[3:] != video.shape[3:]:
            raise ValueError("上一片段与当前片段分辨率不一致，无法直接传递 Latent")
        if previous_video.shape[2] < video_steps:
            raise ValueError("上一片段视频 Latent 短于要求的段间锁定窗口")
        tail_latent = previous_video[:, :, -video_steps:]
    else:
        tail = previous_images[-locked_frames:, ..., :3]
        if tail.shape[1] != height or tail.shape[2] != width:
            tail = resize_h3_image(tail, width, height, "disabled")
        tail_latent = video_vae.encode(tail)
        previous_audio_latent = None
        if tail_latent.shape[2] != video_steps:
            raise ValueError(f"{locked_frames} 帧段间画面编码得到 {tail_latent.shape[2]} 个 Latent 步，预期 {video_steps} 个")
    if video_steps >= video.shape[2]:
        raise ValueError("段间锁定窗口必须短于本次生成视频")
    video[:, :, :video_steps] = tail_latent.to(device=video.device, dtype=video.dtype)
    video_mask = torch.ones_like(video)
    video_mask[:, :, :video_steps] = 0

    audio_mask = torch.ones_like(audio)
    audio_steps = max(1, round((locked_frames / FPS) * 40))
    if previous_audio_latent is not None:
        if previous_audio_latent.shape[-1] < audio_steps:
            raise ValueError("上一片段音频 Latent 短于要求的段间锁定窗口")
        audio_latent = previous_audio_latent[..., -audio_steps:]
    elif previous_audio is not None:
        sample_rate = int(previous_audio["sample_rate"])
        audio_samples = max(1, round((locked_frames / FPS) * sample_rate))
        tail_audio = {**previous_audio, "waveform": previous_audio["waveform"][..., -audio_samples:]}
        audio_latent = _encode_context_audio(audio_vae, tail_audio)
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
    labels = {id(actor): actor.card.name or f"Actor {index}"
        for index, actor in enumerate(timeline.characters.actors, 1)}
    states = []
    for track in timeline.tracks.tracks:
        completed = [clip for clip in track.clips if clip.end_time <= start + 1e-6 and clip.result]
        if not completed:
            continue
        clip = max(completed, key=lambda item: item.end_time)
        owner = labels.get(id(track.owner), "The environment" if track.owner_kind == "environment" else "The scene")
        states.append(_sentence(f"At the beginning of this segment, {owner} remains in this established state: {clip.result}"))
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
                          previous_images=None, previous_audio=None):
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
        visible = _reference_visible_frames(reference, timeline.duration)
        locked = None
        previous_reference = previous.get(reference.owner_id)
        if frame_plan.locked_frames and previous_images is not None and previous_images.shape[0]:
            locked = previous_images[-frame_plan.locked_frames:, ..., :3]
        elif frame_plan.locked_frames and previous_reference is not None:
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
        if visible.shape[0] != target_frames:
            raise ValueError(f"参考视频对齐后得到 {visible.shape[0]} 帧，本次模型生成需要 {target_frames} 帧")

        audio = reference.audio
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            waveform = audio["waveform"]
            channels = waveform.shape[1]
            body = _reference_visible_audio(reference, timeline.duration, sample_rate, channels,
                waveform.device, waveform.dtype)
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
            locked_duration=frame_plan.locked_frames / FPS))
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


def _opening_alignment_instruction(locked_duration):
    if locked_duration <= 1e-6:
        return ""
    return (f"The opening {_time(locked_duration)} seconds are hard-locked to the preceding generated segment's final motion, pose, and framing. "
        f"At {_time(locked_duration)} seconds, the current action continues directly from the locked final frame and runs through the final frame")


def _semantic_references(timeline, video_groups, audio_groups, first_subject_number,
                         prefix_duration=0.0, has_locked_context=False):
    actor_subjects = _reference_subject_layout(timeline)[0]
    labels = {id(actor): actor.card.name or f"Actor {index}"
        for index, actor in enumerate(timeline.characters.actors, 1)}
    instructions = []
    definitions = []
    retentions = []
    action_labels = []
    camera_labels = []
    lighting_labels = []
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
        if prefix_duration or has_camera:
            roles = []
            if prefix_duration:
                roles.append("motion-transition context and current shot-aligned temporal order" if has_locked_context
                    else "current shot-aligned temporal order")
            if has_camera:
                roles.append("camera movement and framing progression")
            definitions.append(f"<Video {video_number}> is the reference for [Shot 1]'s {' and '.join(roles)}.")
            retentions.append(f"<Video {video_number}> (temporal structure in [Shot 1]): fully_preserved - "
                f"Preserve its {' and '.join(roles)} without importing source identities, clothing, or environment.")
        for kind, track, clip, _ in bindings:
            if kind == "actor":
                owner = labels.get(id(track.owner), "the character")
                target = (f"<Subject {actor_subjects[id(track.owner)]}> ({owner})"
                    if id(track.owner) in actor_subjects else owner)
                subject = f"<Subject {subject_number}>"
                definitions.append(f"{subject} is the body performance derived from <Video {video_number}> and transferred to {target}.")
                retentions.append(f"{subject} (appears in [Shot 1]): attribute_transfer - Transfer its motion order, timing, and final pose to {target}.")
                line = (f"Use {subject} as {target}'s authoritative body performance. Preserve its complete motion order, timing, "
                    f"weight shifts, and final pose while retaining {owner}'s declared identity and fixed appearance. "
                    "Transfer only body performance; do not copy the source performer, face, clothing, background, lighting, or audio.")
                instructions.append((id(clip), _sentence(line)))
                action_labels.append(subject)
                subject_number += 1
            elif kind == "camera":
                instructions.append((id(clip), _sentence(
                    f"Use <Video {video_number}> as the authoritative camera movement, framing, and temporal-structure reference. "
                    "Transfer only camera behavior; do not copy the source performer, clothing, background, lighting, or audio.")))
                camera_labels.append(f"<Video {video_number}>")
            elif kind == "lighting":
                subject = f"<Subject {subject_number}>"
                definitions.append(f"{subject} is the lighting behavior derived from <Video {video_number}> and transferred to [Shot 1].")
                retentions.append(f"{subject} (appears in [Shot 1]): attribute_transfer - Transfer its light direction, color, intensity, and change rhythm without importing the source scene.")
                instructions.append((id(clip), _sentence(
                    f"Use {subject} as [Shot 1]'s authoritative lighting behavior while preserving the declared subjects and environment.")))
                lighting_labels.append(subject)
                subject_number += 1
            elif kind == "audio":
                number = audio_numbers[id(group)]
                label = f"<Audio {number}>"
                definitions.append(f"{label} is the {clip.audio_type} sound reference from the soundtrack associated with <Video {video_number}>.")
                retentions.append(f"{label}: reference - Follow its timing and sound character without copying the source signal.")
                instructions.append((id(clip), _sentence(f"Use {label} as the {clip.audio_type} timing and sound-character reference.")))
                audio_labels.append(label)

    standalone_audios = []
    for group in audio_groups:
        binding = group["bindings"][0]
        _, _, clip, source = binding
        number = audio_numbers[id(group)]
        label = f"<Audio {number}>"
        definitions.append(f"{label} is the {clip.audio_type} sound reference from the supplied reference video soundtrack.")
        retentions.append(f"{label}: reference - Follow its timing and sound character without copying the source signal.")
        instructions.append((id(clip), _sentence(f"Use {label} as the {clip.audio_type} timing and sound-character reference.")))
        audio_labels.append(label)
        standalone_audios.append(source.audio)

    instruction_map = {}
    for clip_id, instruction in instructions:
        instruction_map[clip_id] = " ".join(filter(_text, (instruction_map.get(clip_id, ""), instruction)))
    summary_parts = []
    if action_labels:
        summary_parts.append(f"Body performance is transferred from {' and '.join(action_labels)}.")
    if camera_labels:
        summary_parts.append(f"Camera movement and temporal structure follow {' and '.join(dict.fromkeys(camera_labels))}.")
    if lighting_labels:
        summary_parts.append(f"Lighting behavior is transferred from {' and '.join(lighting_labels)}.")
    if audio_labels:
        summary_parts.append(f"Sound is referenced from {' and '.join(dict.fromkeys(audio_labels))}.")
    return instruction_map, "\n".join(definitions), "\n".join(retentions), " ".join(summary_parts), standalone_audios


def _compile_generation_segment(generation_job, segment_index, frame_plan=None,
                                previous_images=None, previous_audio=None):
    ranges = _segment_ranges(generation_job.timeline)
    if segment_index < 0 or segment_index >= len(ranges):
        raise ValueError(f"分段编号超出范围：{segment_index}")
    start, end = ranges[segment_index]
    if frame_plan is None:
        available = round((ranges[segment_index - 1][1] - ranges[segment_index - 1][0]) * FPS) if segment_index else 0
        frame_plan = _segment_frame_plan(generation_job.continuity_seconds, available, round((end - start) * FPS))
    timeline = _retime_timeline(_segment_timeline(generation_job.timeline, start, end),
        frame_plan.current_frames / FPS)
    previous_timeline = (_segment_timeline(generation_job.timeline, *ranges[segment_index - 1])
        if segment_index else None)
    state = _persistent_state(generation_job.timeline, start)
    prefix_duration = frame_plan.locked_frames / FPS
    locked_duration = frame_plan.locked_frames / FPS
    video_groups, audio_groups = _reference_media(timeline)
    media_references = [reference for _, reference in video_groups]
    motion_instructions, motion_definitions, motion_retentions, motion_summary, standalone_audios = _semantic_references(
        timeline, video_groups, audio_groups, _reference_subject_count(timeline) + 1, prefix_duration,
        frame_plan.locked_frames > 0)
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
        suppress_initial_state=frame_plan.locked_frames > 0,
        generation_duration=frame_plan.generation_frames / FPS,
        timeline_offset=prefix_duration,
        opening_instructions=_opening_alignment_instruction(locked_duration),
        continuation=frame_plan.locked_frames > 0,
    )[0]
    if compiled.video_settings.length != frame_plan.generation_frames:
        raise ValueError(f"提示词生成长度为 {compiled.video_settings.length} 帧，分段计划需要 {frame_plan.generation_frames} 帧")
    media_references = _align_motion_context(timeline, previous_timeline, media_references, frame_plan,
        previous_images, previous_audio)
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
        frame_plan = _segment_frame_plan(generation_job.continuity_seconds,
            previous_images.shape[0] if previous_images is not None else 0, round((end - start) * FPS))
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
                video_vae, audio_vae, frame_plan.locked_frames, settings.width, settings.height)
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
            available = round((ranges[index - 1][1] - ranges[index - 1][0]) * 24) if index else 0
            frame_plan = _segment_frame_plan(generation_job.continuity_seconds, available,
                round((end - start) * FPS))
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
                references.insert(0,
                    f"段间引导 = Latent 强锁定上一片段末尾 {frame_plan.locked_frames} 帧及对应音频，"
                    f"生成后只裁掉这 {frame_plan.locked_frames} 帧（不占用参考视频槽位）")
            
            header = [f"========== 片段 {index + 1}/{len(ranges)} ==========",
                f"时间轴范围：{_time(start)}–{_time(end)} 秒 | 动态输出时长：{_time(frame_plan.current_frames / FPS)} 秒"]
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
