from dataclasses import replace

import torch

from comfy_api.latest import io

from .schema import (ACTOR_KINDS, CATEGORY, FPS, H3_ACTOR_INSTANCE, H3_ENVIRONMENT_INSTANCE,
    H3_LANGUAGE, H3_MOTION_REFERENCE, H3_TIMELINE_CLIP, H3_TIMELINE_TRACK, H3_TRACK_LIST,
    SYSTEM_KINDS, ActorInstanceData, LanguageData, MotionReferenceData, TimelineClipData,
    TimelineTrackData, TrackListData)
from .utils import _autogrow, _sentence, _text, _values


def _nearest_reference_frame_count(duration):
    wanted = max(5, round(duration * FPS))
    lower = max(5, 5 + 17 * ((wanted - 5) // 17))
    upper = lower + 17
    maximum = 5 + 17 * ((round(15.0 * FPS) - 5) // 17)
    return min(maximum, lower if wanted - lower <= upper - wanted else upper)


def _align_motion_reference(reference, duration):
    target_frames = _nearest_reference_frame_count(duration)
    source_frames = reference.frames
    positions = torch.linspace(0, source_frames.shape[0] - 1, target_frames,
        device=source_frames.device).round().long()
    frames = source_frames.index_select(0, positions)
    aligned_duration = target_frames / FPS
    audio = reference.audio
    if audio is not None:
        sample_count = max(1, round(aligned_duration * audio["sample_rate"]))
        waveform = audio["waveform"]
        shape = waveform.shape
        waveform = torch.nn.functional.interpolate(waveform.reshape(-1, 1, shape[-1]),
            size=sample_count, mode="linear", align_corners=False).reshape(*shape[:-1], sample_count)
        audio = {**audio, "waveform": waveform}
    source_duration = reference.source_duration or source_frames.shape[0] / FPS
    return MotionReferenceData(frames, audio, reference.role, source_duration, aligned_duration)


class MiniMaxH3MotionReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MotionReference", display_name="MiniMax H3 动作参考视频（Motion Reference）",
            category=CATEGORY, inputs=[io.Video.Input("video", display_name="参考视频"),
            io.Combo.Input("role", display_name="参考范围", options=["仅动作", "动作与镜头", "完整表演", "动作与声音"], default="仅动作"),
            io.Boolean.Input("include_audio", display_name="包含参考音频", default=False),
            io.Float.Input("trim_start", display_name="截取开始（秒）", default=0.0, min=0.0, max=3600.0, step=0.05),
            io.Float.Input("trim_end", display_name="截取结束（秒，0为结尾）", default=0.0, min=0.0, max=3600.0, step=0.05)],
            outputs=[H3_MOTION_REFERENCE.Output(display_name="motion_reference")])

    @classmethod
    def execute(cls, video, role, include_audio, trim_start, trim_end):
        components = video.get_components()
        frames = components.images
        source_fps = float(components.frame_rate)
        source_duration = frames.shape[0] / source_fps
        end = source_duration if trim_end <= 0.0 else min(trim_end, source_duration)
        if trim_start >= end:
            raise ValueError("动作参考视频的截取开始时间必须早于结束时间")
        duration = end - trim_start
        if duration > 15.0 + 1e-6:
            raise ValueError("MiniMax H3 的单个动作参考视频不能超过 15 秒，请先截取")
        target_frames = max(5, round(duration * FPS))
        positions = torch.arange(target_frames, device=frames.device, dtype=torch.float32) * (source_fps / FPS)
        indices = (positions + trim_start * source_fps).round().long().clamp(max=frames.shape[0] - 1)
        frames = frames.index_select(0, indices)
        audio = components.audio if include_audio else None
        if audio is not None:
            sample_rate = audio["sample_rate"]
            first_sample = round(trim_start * sample_rate)
            last_sample = min(audio["waveform"].shape[-1], round(end * sample_rate))
            audio = {**audio, "waveform": audio["waveform"][..., first_sample:last_sample]}
        return io.NodeOutput(MotionReferenceData(frames, audio, role, duration, frames.shape[0] / FPS))


class MiniMaxH3Action(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Action", display_name="MiniMax H3 人物动作片段（Actor Action Clip）", category=CATEGORY, inputs=[
            io.Combo.Input("action_type", display_name="动作种类", options=list(ACTOR_KINDS), default="body"),
            io.Float.Input("start_time", display_name="开始时间（秒）", default=0.0, min=0.0, max=60.0, step=0.05),
            io.Float.Input("end_time", display_name="结束时间（秒）", default=1.0, min=0.0, max=60.0, step=0.05),
            io.String.Input("content", display_name="动作内容", placeholder="动作内容", default="raises her right hand naturally", multiline=True),
            io.Combo.Input("speech_type", display_name="说话类型", options=["on-screen", "off-screen voiceover"], default="on-screen"),
            io.String.Input("quality", display_name="动作质量", placeholder="动作质量", default="The movement is physically natural and controlled.", multiline=True, optional=True),
            io.String.Input("result", display_name="结束状态", placeholder="结束状态", default="She keeps her right hand raised.", multiline=True, optional=True),
            io.String.Input("delivery", display_name="说话方式", placeholder="说话方式", default="", multiline=True, optional=True),
            H3_LANGUAGE.Input("language", optional=True), H3_ACTOR_INSTANCE.Input("target", optional=True),
            H3_MOTION_REFERENCE.Input("motion_reference", display_name="动作参考视频", optional=True)],
            outputs=[H3_TIMELINE_CLIP.Output(display_name="clip")])

    @classmethod
    def execute(cls, action_type, start_time, end_time, content, quality="", result="", delivery="", speech_type="on-screen",
                language=None, target=None, motion_reference=None):
        if action_type == "speech" and not isinstance(language, LanguageData):
            raise ValueError("A speech action requires a Language input")
        if action_type != "speech" and language is not None:
            raise ValueError("Language can only be connected to a speech action")
        if target is not None and not isinstance(target, ActorInstanceData):
            raise TypeError("Action target must be an actor instance")
        if motion_reference is not None and not isinstance(motion_reference, MotionReferenceData):
            raise TypeError("动作参考必须来自 MiniMax H3 动作参考视频节点")
        if motion_reference is not None:
            if end_time <= start_time:
                raise ValueError("带动作参考视频的动作片段结束时间必须晚于开始时间")
            motion_reference = _align_motion_reference(motion_reference, end_time - start_time)
        return io.NodeOutput(TimelineClipData(action_type, start_time, end_time, _text(content), _sentence(quality),
            _sentence(result), language, _text(delivery), speech_type, target, "", motion_reference))


class MiniMaxH3ActionResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3ActionResult", display_name="MiniMax H3 动作片段结果扩展（Use Result）",
            category=CATEGORY, description="给动作片段绑定已经生成的视频；命中该片段时跳过模型采样。",
            inputs=[H3_TIMELINE_CLIP.Input("clip", display_name="动作片段"),
                io.Video.Input("video", display_name="已生成结果"),
                io.Int.Input("result_version", display_name="结果版本", default=0, min=0, max=1000000,
                    tooltip="替换输入视频后递增此值，使当前片段及后续连续片段不再复用旧缓存。")],
            outputs=[H3_TIMELINE_CLIP.Output(display_name="clip")])

    @classmethod
    def execute(cls, clip, video, result_version):
        if not isinstance(clip, TimelineClipData) or clip.kind not in ACTOR_KINDS:
            raise TypeError("动作片段结果扩展只能连接人物动作片段")
        return io.NodeOutput(replace(clip, rendered_video=video, rendered_video_version=int(result_version)))


class MiniMaxH3Camera(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Camera", display_name="MiniMax H3 摄像机动作片段（Camera Clip）", category=CATEGORY, inputs=[
            io.Float.Input("start_time", display_name="开始时间（秒）", default=0.0, min=0.0, max=60.0, step=0.05),
            io.Float.Input("end_time", display_name="结束时间（秒）", default=5.0, min=0.0, max=60.0, step=0.05),
            io.String.Input("framing_and_angle", display_name="景别与角度", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="The camera begins at eye level in a medium shot, keeping {actor_0} centered.", multiline=True),
            io.String.Input("movement", display_name="摄像机运动", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="It slowly pushes toward {actor_0}.", multiline=True),
            io.String.Input("focus", display_name="对焦与景深", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="A shallow depth of field keeps {actor_0}'s face sharp.", multiline=True),
            io.String.Input("result", display_name="结束状态", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="The camera holds on the final framing.", multiline=True, optional=True)],
            outputs=[H3_TIMELINE_CLIP.Output(display_name="clip")])

    @classmethod
    def execute(cls, start_time, end_time, framing_and_angle, movement, focus, result=""):
        content = " ".join(map(_sentence, filter(_text, (framing_and_angle, movement, focus))))
        return io.NodeOutput(TimelineClipData("camera", start_time, end_time, content, "", _sentence(result)))


class MiniMaxH3LightingAction(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LightingAction", display_name="MiniMax H3 灯光动作片段（Lighting Clip）", category=CATEGORY, inputs=[
            io.Float.Input("start_time", display_name="开始时间（秒）", default=0.0, min=0.0, max=60.0, step=0.05),
            io.Float.Input("end_time", display_name="结束时间（秒）", default=5.0, min=0.0, max=60.0, step=0.05),
            io.String.Input("lighting", display_name="灯光描述", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="Cool city light separates {actor_0} from the background.", multiline=True),
            io.String.Input("transition", display_name="灯光变化", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="The lighting remains stable without flicker.", multiline=True, optional=True),
            io.String.Input("result", display_name="结束状态", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="The final lighting state remains stable.", multiline=True, optional=True)],
            outputs=[H3_TIMELINE_CLIP.Output(display_name="clip")])

    @classmethod
    def execute(cls, start_time, end_time, lighting, transition="", result=""):
        return io.NodeOutput(TimelineClipData("lighting", start_time, end_time, _text(lighting), _sentence(transition), _sentence(result)))


class MiniMaxH3AudioAction(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3AudioAction", display_name="MiniMax H3 音频片段（Audio Clip）", category=CATEGORY, inputs=[
            io.Combo.Input("audio_type", display_name="音频种类", options=["ambient", "sound effect", "music", "off-screen sound"], default="ambient"),
            io.Float.Input("start_time", display_name="开始时间（秒）", default=0.0, min=0.0, max=60.0, step=0.05),
            io.Float.Input("end_time", display_name="结束时间（秒）", default=5.0, min=0.0, max=60.0, step=0.05),
            io.String.Input("description", display_name="声音描述", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="Rain, wind, and distant traffic remain naturally audible.", multiline=True),
            io.String.Input("volume_and_space", display_name="音量与空间", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="The sound remains soft and spatially coherent.", multiline=True, optional=True),
            io.String.Input("fade", display_name="淡入淡出", placeholder="用 {actor_0}、{actor_1} 引用人物组成员", default="", multiline=True, optional=True)], outputs=[H3_TIMELINE_CLIP.Output(display_name="clip")])

    @classmethod
    def execute(cls, audio_type, start_time, end_time, description, volume_and_space="", fade=""):
        quality = " ".join(filter(_text, map(_sentence, (volume_and_space, fade))))
        return io.NodeOutput(TimelineClipData("audio", start_time, end_time, _text(description), quality, "", audio_type=audio_type))


class MiniMaxH3EnvironmentAction(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3EnvironmentAction", display_name="MiniMax H3 环境动作片段（Environment Clip）", category=CATEGORY, inputs=[
            io.Float.Input("start_time", display_name="开始时间（秒）", default=0.0, min=0.0, max=60.0, step=0.05),
            io.Float.Input("end_time", display_name="结束时间（秒）", default=5.0, min=0.0, max=60.0, step=0.05),
            io.String.Input("change", display_name="环境变化", placeholder="环境变化", default="The rain gradually becomes heavier.", multiline=True),
            io.String.Input("quality", display_name="变化质量", placeholder="变化质量", default="The change occurs continuously and naturally.", multiline=True, optional=True),
            io.String.Input("result", display_name="结束状态", placeholder="结束状态", default="Heavy rain continues.", multiline=True, optional=True)],
            outputs=[H3_TIMELINE_CLIP.Output(display_name="clip")])

    @classmethod
    def execute(cls, start_time, end_time, change, quality="", result=""):
        return io.NodeOutput(TimelineClipData("environment", start_time, end_time, _text(change), _sentence(quality), _sentence(result)))


def _track_clips(clips, allowed, label):
    values = _values(clips)
    for clip in values:
        if not isinstance(clip, TimelineClipData) or clip.kind not in allowed:
            raise ValueError(f"{label} received an incompatible clip")
    return tuple(values)


class MiniMaxH3ActorTrack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3ActorTrack", display_name="MiniMax H3 人物轨道（Actor Track）", category=CATEGORY,
            inputs=[H3_ACTOR_INSTANCE.Input("actor"), _autogrow(H3_TIMELINE_CLIP, "clips", "clip", 0)],
            outputs=[H3_TIMELINE_TRACK.Output(display_name="track")])

    @classmethod
    def execute(cls, actor, clips=None):
        return io.NodeOutput(TimelineTrackData("actor", actor, _track_clips(clips, ACTOR_KINDS, "Actor track")))


class MiniMaxH3EnvironmentTrack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3EnvironmentTrack", display_name="MiniMax H3 环境轨道（Environment Track）", category=CATEGORY,
            inputs=[H3_ENVIRONMENT_INSTANCE.Input("environment"), _autogrow(H3_TIMELINE_CLIP, "clips", "clip", 0)],
            outputs=[H3_TIMELINE_TRACK.Output(display_name="track")])

    @classmethod
    def execute(cls, environment, clips=None):
        return io.NodeOutput(TimelineTrackData("environment", environment, _track_clips(clips, ("environment",), "Environment track")))


class MiniMaxH3SystemTrack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SystemTrack", display_name="MiniMax H3 系统轨道（System Track）", category=CATEGORY,
            inputs=[_autogrow(H3_TIMELINE_CLIP, "clips", "clip", 1)], outputs=[H3_TIMELINE_TRACK.Output(display_name="track")])

    @classmethod
    def execute(cls, clips):
        clips = _track_clips(clips, SYSTEM_KINDS, "System track")
        if len({clip.kind for clip in clips}) != 1:
            raise ValueError("A system track can contain only one clip kind")
        return io.NodeOutput(TimelineTrackData(clips[0].kind, None, clips))


class MiniMaxH3TrackList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3TrackList", display_name="MiniMax H3 轨道数组（Track List）", category=CATEGORY,
            inputs=[_autogrow(H3_TIMELINE_TRACK, "tracks", "track", 0)], outputs=[H3_TRACK_LIST.Output(display_name="tracks")])

    @classmethod
    def execute(cls, tracks=None):
        tracks = _values(tracks)
        if any(not isinstance(track, TimelineTrackData) for track in tracks):
            raise TypeError("Track list accepts only timeline tracks")
        return io.NodeOutput(TrackListData(tuple(tracks)))

