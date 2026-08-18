import math
import re
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo as NativeRef2VA
from typing_extensions import override


CATEGORY = "MiniMax H3/提示词构建"
WEB_DIRECTORY = "./web"
FPS = 24
ASPECT_RATIOS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "21:9": (21, 9)}
ACTOR_KINDS = ("body", "expression", "gaze", "speech")
SYSTEM_KINDS = ("camera", "lighting", "audio")

H3_CHARACTER_CARD = io.Custom("MINIMAX_H3_CHARACTER_CARD")
H3_ACTOR_INSTANCE = io.Custom("MINIMAX_H3_ACTOR_INSTANCE")
H3_CHARACTER_GROUP = io.Custom("MINIMAX_H3_CHARACTER_GROUP")
H3_LANGUAGE = io.Custom("MINIMAX_H3_LANGUAGE")
H3_STYLE_CARD = io.Custom("MINIMAX_H3_STYLE_CARD")
H3_ENVIRONMENT_CARD = io.Custom("MINIMAX_H3_ENVIRONMENT_CARD")
H3_ENVIRONMENT_INSTANCE = io.Custom("MINIMAX_H3_ENVIRONMENT_INSTANCE")
H3_MOTION_REFERENCE = io.Custom("MINIMAX_H3_MOTION_REFERENCE")
H3_TIMELINE_CLIP = io.Custom("MINIMAX_H3_TIMELINE_CLIP")
H3_TIMELINE_TRACK = io.Custom("MINIMAX_H3_TIMELINE_TRACK")
H3_TRACK_LIST = io.Custom("MINIMAX_H3_TRACK_LIST")
H3_TIMELINE = io.Custom("MINIMAX_H3_TIMELINE")
H3_PROMPT = io.Custom("MINIMAX_H3_PROMPT")
H3_GENERATION_JOB = io.Custom("MINIMAX_H3_GENERATION_JOB")


@dataclass(frozen=True, slots=True)
class ReferenceImageData:
    picture_number: int
    image: object
    role: str
    usage: str


@dataclass(frozen=True, slots=True)
class CharacterCardData:
    name: str
    description: str
    preservation: str
    reference: ReferenceImageData | None
    default_position: str
    default_pose: str
    default_emotion: str
    default_appearance: str
    character_style: str
    style_priority: str


@dataclass(frozen=True, slots=True)
class ActorInstanceData:
    card: CharacterCardData
    position_override: str
    pose_override: str
    emotion_override: str
    appearance_override: str


@dataclass(frozen=True, slots=True)
class CharacterGroupData:
    actors: tuple[ActorInstanceData, ...]


@dataclass(frozen=True, slots=True)
class LanguageData:
    language: str
    variant: str
    accent: str
    pronunciation: str


@dataclass(frozen=True, slots=True)
class StyleCardData:
    style: str
    rendering: str
    color_palette: str
    texture: str
    reference: ReferenceImageData | None


@dataclass(frozen=True, slots=True)
class EnvironmentCardData:
    name: str
    location: str
    default_time_weather: str
    default_background: str
    default_atmosphere: str
    preservation: str
    reference: ReferenceImageData | None


@dataclass(frozen=True, slots=True)
class EnvironmentInstanceData:
    card: EnvironmentCardData
    location_override: str
    time_weather_override: str
    background_override: str
    atmosphere_override: str


@dataclass(frozen=True, slots=True, eq=False)
class MotionReferenceData:
    frames: object
    audio: object | None
    role: str


@dataclass(frozen=True, slots=True)
class TimelineClipData:
    kind: str
    start_time: float
    end_time: float
    content: str
    quality: str
    result: str
    language: LanguageData | None = None
    delivery: str = ""
    speech_type: str = "on-screen"
    target: ActorInstanceData | None = None
    audio_type: str = ""
    motion_reference: MotionReferenceData | None = None


@dataclass(frozen=True, slots=True)
class TimelineTrackData:
    owner_kind: str
    owner: object | None
    clips: tuple[TimelineClipData, ...]


@dataclass(frozen=True, slots=True)
class TrackListData:
    tracks: tuple[TimelineTrackData, ...]


@dataclass(frozen=True, slots=True)
class TimelineData:
    characters: CharacterGroupData
    style: StyleCardData
    environment: EnvironmentInstanceData
    tracks: TrackListData
    duration: float


@dataclass(frozen=True, slots=True)
class VideoSettingsData:
    mode: str
    width: int
    height: int
    length: int
    duration: float
    first_frame: object | None
    last_frame: object | None


@dataclass(frozen=True, slots=True)
class CompletePromptData:
    text: str
    references: tuple[ReferenceImageData, ...]
    video_settings: VideoSettingsData


@dataclass(frozen=True, slots=True)
class GenerationJobData:
    timeline: TimelineData
    megapixels: float
    aspect_ratio: str
    seed: int
    scheduler: str
    steps: int
    denoise: float
    ref_image_size: str
    continuity_seconds: float
    overlap_seconds: float


for custom_type, data_type in ((H3_CHARACTER_CARD, CharacterCardData), (H3_ACTOR_INSTANCE, ActorInstanceData),
                               (H3_CHARACTER_GROUP, CharacterGroupData), (H3_LANGUAGE, LanguageData),
                               (H3_STYLE_CARD, StyleCardData), (H3_ENVIRONMENT_CARD, EnvironmentCardData),
                               (H3_ENVIRONMENT_INSTANCE, EnvironmentInstanceData), (H3_MOTION_REFERENCE, MotionReferenceData),
                               (H3_TIMELINE_CLIP, TimelineClipData),
                               (H3_TIMELINE_TRACK, TimelineTrackData), (H3_TRACK_LIST, TrackListData),
                               (H3_TIMELINE, TimelineData), (H3_PROMPT, CompletePromptData),
                               (H3_GENERATION_JOB, GenerationJobData)):
    custom_type.Type = data_type


def _text(value):
    return value.strip() if value else ""


def _sentence(value):
    value = _text(value)
    target = value[:-4].rstrip() if value.endswith("</d>") else value
    return value + "." if value and (not target or target[-1] not in ".!?。！？") else value


def _values(autogrow):
    return list(autogrow.values()) if autogrow else []


def _autogrow(input_type, name, prefix, minimum, maximum=100):
    return io.Autogrow.Input(name, optional=minimum == 0, template=io.Autogrow.TemplatePrefix(
        input=input_type.Input(prefix), prefix=f"{prefix}_", min=minimum, max=maximum,
    ))


def _reference(image, role, usage=""):
    return ReferenceImageData(0, image, role, _text(usage)) if image is not None else None


def _same_image(left, right):
    return left is right or (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                             and left.shape == right.shape and torch.equal(left, right))


def _resolved(default, override):
    return _sentence(override) or _sentence(default)


def _actor_state(label, default, override):
    state = _text(override) or _text(default)
    if not state:
        return ""
    return _sentence(f"{label} {state}")


def _bind_actor_tokens(value, actor_labels):
    value = _text(value)
    if not value:
        return ""

    def replace_actor(match):
        socket_suffix = match.group(1)
        if not re.fullmatch(r"_\d+", socket_suffix):
            raise ValueError(f"Invalid actor placeholder {match.group(0)}; use {{actor_0}}, {{actor_1}}, and so on")
        index = int(socket_suffix[1:])
        if index >= len(actor_labels):
            raise ValueError(f"{match.group(0)} exceeds the Character Group, which contains {len(actor_labels)} actor(s)")
        return actor_labels[index]

    return re.sub(r"\{actor([^}]*)\}", replace_actor, value)


def _video_size(megapixels, aspect_ratio):
    ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
    scale = math.sqrt(megapixels * 1024 * 1024 / (ratio_width * ratio_height))
    return round(ratio_width * scale / 32) * 32, round(ratio_height * scale / 32) * 32


def _video_length(duration):
    frames = max(5, math.ceil(duration * FPS))
    remainder = (frames - 5) % 17
    return frames if not remainder else frames + 17 - remainder


def _match_reference_video(frames, width, height):
    source_height, source_width = frames.shape[1:3]
    scale = min(1.0, math.sqrt((width * height) / (source_width * source_height)))
    target_width = max(32, round(source_width * scale / 32) * 32)
    target_height = max(32, round(source_height * scale / 32) * 32)
    if (target_height, target_width) == (source_height, source_width):
        return frames
    samples = frames[..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, target_width, target_height, "lanczos", "disabled").movedim(1, -1)


def _time(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


class MiniMaxH3Character(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Character", display_name="MiniMax H3 人物卡（Character Card）", category=CATEGORY, inputs=[
            io.String.Input("name", display_name="人物名称", placeholder="人物名称", default="the young woman"),
            io.String.Input("description", display_name="人物描述", placeholder="人物描述", default="A young woman with long black hair wearing a dark red coat.", multiline=True),
            io.String.Input("preservation", display_name="一致性要求", placeholder="一致性要求", default="Preserve her identity and appearance throughout the video.", multiline=True),
            io.String.Input("default_position", display_name="默认位置", placeholder="只写状态，例如：站在画面中央", default="stands in the center of the frame", multiline=True),
            io.String.Input("default_pose", display_name="默认姿态", placeholder="只写状态，例如：自然放松地站立", default="stands naturally with a relaxed posture", multiline=True),
            io.String.Input("default_emotion", display_name="默认表情", placeholder="只写状态，例如：神情平静", default="has a calm expression", multiline=True),
            io.Combo.Input("style_priority", display_name="风格优先级", options=["character", "global"], default="global"),
            io.String.Input("default_appearance", display_name="默认外观", placeholder="默认外观", default="", multiline=True, optional=True),
            io.String.Input("character_style", display_name="人物风格", placeholder="人物风格", default="", multiline=True, optional=True),
            io.Image.Input("reference_image", optional=True),
        ], outputs=[H3_CHARACTER_CARD.Output(display_name="character_card")])

    @classmethod
    def execute(cls, name, description, preservation, default_position, default_pose, default_emotion,
                default_appearance, character_style, style_priority, reference_image=None):
        return io.NodeOutput(CharacterCardData(_text(name) or "the character", _sentence(description), _sentence(preservation),
            _reference(reference_image, "character identity"), _sentence(default_position), _sentence(default_pose),
            _sentence(default_emotion), _sentence(default_appearance), _sentence(character_style), style_priority))


class MiniMaxH3ActorInstance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3ActorInstance", display_name="MiniMax H3 人物实例（Actor Instance）", category=CATEGORY,
            description="空字段继承人物卡，非空字段覆盖人物卡。", inputs=[H3_CHARACTER_CARD.Input("character_card"),
            io.String.Input("position_override", display_name="位置覆盖", placeholder="位置覆盖", default="", multiline=True, optional=True),
            io.String.Input("pose_override", display_name="姿态覆盖", placeholder="姿态覆盖", default="", multiline=True, optional=True),
            io.String.Input("emotion_override", display_name="表情覆盖", placeholder="表情覆盖", default="", multiline=True, optional=True),
            io.String.Input("appearance_override", display_name="外观覆盖", placeholder="外观覆盖", default="", multiline=True, optional=True)],
            outputs=[H3_ACTOR_INSTANCE.Output(display_name="actor_instance")])

    @classmethod
    def execute(cls, character_card, position_override="", pose_override="", emotion_override="", appearance_override=""):
        if not isinstance(character_card, CharacterCardData):
            raise TypeError("Actor instance requires a character card")
        return io.NodeOutput(ActorInstanceData(character_card, *map(_text, (position_override, pose_override, emotion_override, appearance_override))))


class MiniMaxH3CharacterGroup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3CharacterGroup", display_name="MiniMax H3 人物组（Character Group）", category=CATEGORY,
            description="按连接顺序自动分配 S1、S2……", inputs=[_autogrow(H3_ACTOR_INSTANCE, "actors", "actor", 1)],
            outputs=[H3_CHARACTER_GROUP.Output(display_name="character_group")])

    @classmethod
    def execute(cls, actors):
        actors = _values(actors)
        if any(not isinstance(actor, ActorInstanceData) for actor in actors):
            raise TypeError("Character group accepts only actor instances")
        if len({id(actor) for actor in actors}) != len(actors):
            raise ValueError("The same actor instance is declared more than once")
        return io.NodeOutput(CharacterGroupData(tuple(actors)))


class MiniMaxH3Language(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Language", display_name="MiniMax H3 语言（Language）", category=CATEGORY, inputs=[
            io.String.Input("language", display_name="语言", placeholder="语言", default="Chinese"), io.String.Input("variant", display_name="语种变体", placeholder="语种变体", default="Mandarin Chinese"),
            io.String.Input("accent", display_name="口音", placeholder="口音", default="standard Mandarin accent"),
            io.String.Input("pronunciation", display_name="发音要求", placeholder="发音要求", default="natural pronunciation with clear articulation", multiline=True)],
            outputs=[H3_LANGUAGE.Output(display_name="language")])

    @classmethod
    def execute(cls, language, variant, accent, pronunciation):
        return io.NodeOutput(LanguageData(_text(language) or "Chinese", _text(variant), _text(accent), _text(pronunciation)))


class MiniMaxH3Visual(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Visual", display_name="MiniMax H3 风格卡（Style Card）", category=CATEGORY, inputs=[
            io.String.Input("style", display_name="视觉风格", placeholder="视觉风格", default="Live-action cinematic realism.", multiline=True),
            io.String.Input("rendering", display_name="渲染表现", placeholder="渲染表现", default="Natural materials and physically coherent motion.", multiline=True),
            io.String.Input("color_palette", display_name="色彩方案", placeholder="色彩方案", default="A restrained cinematic color palette.", multiline=True),
            io.String.Input("texture", display_name="画面质感", placeholder="画面质感", default="Fine, stable image detail without flicker.", multiline=True),
            io.String.Input("reference_usage", display_name="参考图用途", placeholder="参考图用途", default="Use the picture for visual style only without copying subject identity.", multiline=True, advanced=True),
            io.Image.Input("reference_image", optional=True)], outputs=[H3_STYLE_CARD.Output(display_name="style_card")])

    @classmethod
    def execute(cls, style, rendering, color_palette, texture, reference_usage, reference_image=None):
        return io.NodeOutput(StyleCardData(*map(_sentence, (style, rendering, color_palette, texture)),
            _reference(reference_image, "visual style", reference_usage)))


class MiniMaxH3Environment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Environment", display_name="MiniMax H3 环境卡（Environment Card）", category=CATEGORY, inputs=[
            io.String.Input("name", display_name="环境名称", placeholder="环境名称", default="city pedestrian bridge"),
            io.String.Input("location", display_name="地点描述", placeholder="地点描述", default="The scene takes place on a rain-soaked pedestrian bridge.", multiline=True),
            io.String.Input("default_time_weather", display_name="默认时间与天气", placeholder="默认时间与天气", default="It is a windy night with steady rain.", multiline=True),
            io.String.Input("default_background", display_name="默认背景", placeholder="默认背景", default="Distant traffic and neon signs remain visible.", multiline=True),
            io.String.Input("default_atmosphere", display_name="默认氛围", placeholder="默认氛围", default="The atmosphere is quiet and intimate.", multiline=True),
            io.String.Input("preservation", display_name="一致性要求", placeholder="一致性要求", default="Preserve the location and spatial layout.", multiline=True),
            io.String.Input("reference_usage", display_name="参考图用途", placeholder="参考图用途", default="Use the picture for location and spatial layout only.", multiline=True, advanced=True),
            io.Image.Input("reference_image", optional=True)], outputs=[H3_ENVIRONMENT_CARD.Output(display_name="environment_card")])

    @classmethod
    def execute(cls, name, location, default_time_weather, default_background, default_atmosphere, preservation, reference_usage, reference_image=None):
        return io.NodeOutput(EnvironmentCardData(_text(name) or "the environment", *map(_sentence,
            (location, default_time_weather, default_background, default_atmosphere, preservation)),
            _reference(reference_image, "environment", reference_usage)))


class MiniMaxH3EnvironmentInstance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3EnvironmentInstance", display_name="MiniMax H3 环境实例（Environment Instance）", category=CATEGORY,
            description="空字段继承环境卡，非空字段覆盖环境卡。", inputs=[H3_ENVIRONMENT_CARD.Input("environment_card"),
            io.String.Input("location_override", display_name="地点覆盖", placeholder="地点覆盖", default="", multiline=True, optional=True),
            io.String.Input("time_weather_override", display_name="时间与天气覆盖", placeholder="时间与天气覆盖", default="", multiline=True, optional=True),
            io.String.Input("background_override", display_name="背景覆盖", placeholder="背景覆盖", default="", multiline=True, optional=True),
            io.String.Input("atmosphere_override", display_name="氛围覆盖", placeholder="氛围覆盖", default="", multiline=True, optional=True)],
            outputs=[H3_ENVIRONMENT_INSTANCE.Output(display_name="environment_instance")])

    @classmethod
    def execute(cls, environment_card, location_override="", time_weather_override="", background_override="", atmosphere_override=""):
        if not isinstance(environment_card, EnvironmentCardData):
            raise TypeError("Environment instance requires an environment card")
        return io.NodeOutput(EnvironmentInstanceData(environment_card, *map(_text,
            (location_override, time_weather_override, background_override, atmosphere_override))))


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
        return io.NodeOutput(MotionReferenceData(frames, audio, role))


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
        return io.NodeOutput(TimelineClipData(action_type, start_time, end_time, _text(content), _sentence(quality),
            _sentence(result), language, _text(delivery), speech_type, target, "", motion_reference))


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


class MiniMaxH3Timeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Timeline", display_name="MiniMax H3 总时间轴（Timeline）", category=CATEGORY, inputs=[
            H3_CHARACTER_GROUP.Input("character_group"), H3_STYLE_CARD.Input("style_card"),
            H3_ENVIRONMENT_INSTANCE.Input("environment"), H3_TRACK_LIST.Input("tracks"),
            io.Float.Input("duration_seconds", display_name="时长（秒）", default=5.0, min=0.21, max=60.0, step=0.05)],
            outputs=[H3_TIMELINE.Output(display_name="timeline")])

    @classmethod
    def execute(cls, character_group, style_card, environment, tracks, duration_seconds):
        return io.NodeOutput(TimelineData(character_group, style_card, environment, tracks, duration_seconds))


def _validate_timeline(timeline):
    actor_ids = {id(actor) for actor in timeline.characters.actors}
    intervals = {}
    for track_index, track in enumerate(timeline.tracks.tracks, 1):
        if track.owner_kind == "actor":
            if id(track.owner) not in actor_ids:
                raise ValueError(f"Actor track {track_index} references an undeclared actor")
            owner = ("actor", id(track.owner))
        elif track.owner_kind == "environment":
            if track.owner is not timeline.environment:
                raise ValueError(f"Environment track {track_index} uses another environment instance")
            owner = ("environment", id(track.owner))
        else:
            owner = ("system", track.owner_kind)
        for clip_index, clip in enumerate(track.clips, 1):
            if clip.start_time < 0 or clip.end_time <= clip.start_time:
                raise ValueError(f"{track.owner_kind} clip {clip_index} must satisfy 0 <= start_time < end_time")
            if clip.end_time > timeline.duration + 1e-6:
                raise ValueError(f"{track.owner_kind} clip {clip_index} exceeds the timeline duration")
            if clip.target is not None and id(clip.target) not in actor_ids:
                raise ValueError(f"{track.owner_kind} clip {clip_index} references an undeclared actor")
            if clip.kind != "audio":
                intervals.setdefault((owner, clip.kind), []).append(clip)
    for (owner, kind), clips in intervals.items():
        clips.sort(key=lambda clip: (clip.start_time, clip.end_time))
        for previous, current in zip(clips, clips[1:]):
            if current.start_time < previous.end_time - 1e-6:
                raise ValueError(f"Timeline conflict: {owner[0]} {kind} clips overlap")


def _render_clip(track, clip, labels):
    prefix = f"From {_time(clip.start_time)} to {_time(clip.end_time)} seconds, "
    content = clip.content
    quality = clip.quality
    end_state = clip.result
    if track.owner_kind == "actor":
        label = labels[id(track.owner)]
        end_state = _actor_state(label, clip.result, "")
        if clip.kind == "speech":
            details = ", ".join(filter(None, (clip.language.variant, clip.language.accent, clip.language.pronunciation)))
            voice = f" using {details}" if details else ""
            delivery = f" {_text(clip.delivery)}" if clip.delivery else ""
            mode = " says in an off-screen voiceover" if clip.speech_type == "off-screen voiceover" else " says"
            text = f"{label}{mode}{voice}{delivery}: <d>[{clip.language.language}] {content}</d>"
        else:
            text = f"{label} {content}"
            if clip.target is not None:
                if id(clip.target) not in labels:
                    raise ValueError(f"{label}'s action targets an undeclared actor")
                text += f" in relation to {labels[id(clip.target)]}"
    elif track.owner_kind == "environment":
        text = f"the environment changes: {content}"
    elif clip.kind == "lighting":
        text = f"the lighting changes: {content}"
    elif clip.kind == "audio":
        text = f"the {clip.audio_type} sound is heard: {content}"
    else:
        text = content
    result = prefix + _sentence(text)
    if quality:
        result += " " + quality
    if end_state:
        state_name = "post-speech state" if clip.kind == "speech" else f"{clip.kind} state"
        result += " " + _sentence(f"At {_time(clip.end_time)} seconds, {end_state} This resulting {state_name} persists until the next {clip.kind} action")
    return result


class MiniMaxH3FinalPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FinalPrompt", display_name="MiniMax H3 REF 时间轴编译（REF Timeline Compiler）", category=CATEGORY, inputs=[
            H3_TIMELINE.Input("timeline"),
            io.Float.Input("megapixels", display_name="百万像素", default=0.98, min=0.01, max=16.0, step=0.01),
            io.Combo.Input("aspect_ratio", display_name="宽高比", options=list(ASPECT_RATIOS), default="16:9"),
            io.Combo.Input("prompt_format", display_name="提示词模式", options=["Ref", "FL"], default="Ref"),
            io.Image.Input("first_frame", display_name="首帧", optional=True), io.Image.Input("last_frame", display_name="尾帧", optional=True),
            io.String.Input("additional_instructions", display_name="附加要求", placeholder="附加要求", default="", multiline=True, optional=True)],
            outputs=[H3_PROMPT.Output(display_name="complete_prompt")])

    @classmethod
    def execute(cls, timeline, megapixels, aspect_ratio, prompt_format="Ref", first_frame=None, last_frame=None, additional_instructions=""):
        if not isinstance(timeline, TimelineData):
            raise TypeError("Timeline compiler requires TimelineData")
        _validate_timeline(timeline)
        if prompt_format == "Ref" and (first_frame is not None or last_frame is not None):
            raise ValueError("Ref mode does not use first_frame or last_frame")
        width, height = _video_size(megapixels, aspect_ratio)
        length = _video_length(timeline.duration)
        settings = VideoSettingsData("Ref2VA" if prompt_format == "Ref" else "I2VA-continuation", width, height, length,
                                     length / FPS, first_frame, last_frame)
        references, definitions, retentions = [], {}, {}

        def add_reference(reference, definition, marker, retention):
            if reference is None:
                return None
            existing = next((item for item in references if _same_image(item.image, reference.image)), None)
            if existing is None:
                number = len(references) + 1
                references.append(ReferenceImageData(number, reference.image, reference.role, reference.usage))
            else:
                number = existing.picture_number
            definitions.setdefault(number, definition.format(number=number))
            retentions.setdefault(number, (marker, _text(retention)))
            return number

        actor_labels = tuple(f"{actor.card.name} (S{index})" for index, actor in enumerate(timeline.characters.actors, 1))
        labels = {id(actor): actor_labels[index] for index, actor in enumerate(timeline.characters.actors)}
        character_lines = []
        for actor in timeline.characters.actors:
            card, label = actor.card, labels[id(actor)]
            number = None
            if prompt_format == "Ref":
                number = add_reference(card.reference,
                    f"<Subject {{number}}> is {card.name}, whose identity and appearance come from <Picture {{number}}>. {card.description}",
                    "fully_preserved", card.preservation)
            character_lines.append(_sentence(f"{label} is <Subject {number}>" if number else f"{label} is {card.description}"))
            character_lines.extend(filter(_text, (_actor_state(label, card.default_position, actor.position_override),
                _actor_state(label, card.default_pose, actor.pose_override), _actor_state(label, card.default_emotion, actor.emotion_override),
                _actor_state(label, card.default_appearance, actor.appearance_override))))
            if card.character_style:
                rule = "prioritize this character-specific style over conflicting global style" if card.style_priority == "character" else "the global style takes priority; use this character style only where compatible"
                character_lines.append(_sentence(f"For {label}, {rule}: {card.character_style}"))

        style_number = None
        environment_number = None
        if prompt_format == "Ref":
            style_number = add_reference(timeline.style.reference,
                "<Subject {number}> is the visual style derived from <Picture {number}>.", "weak_reference",
                timeline.style.reference.usage if timeline.style.reference else "")
            environment_number = add_reference(timeline.environment.card.reference,
                f"<Subject {{number}}> is the {timeline.environment.card.name} environment and spatial layout derived from <Picture {{number}}>.",
                "partially_preserved", timeline.environment.card.preservation)
        if prompt_format == "Ref" and not references:
            raise ValueError("Ref mode requires at least one character, style, or environment reference image")

        style_parts = [timeline.style.style, timeline.style.rendering, timeline.style.color_palette, timeline.style.texture]
        if style_number:
            style_parts.append(_sentence(f"Use <Subject {style_number}> as the global visual reference. {timeline.style.reference.usage}"))
        card, environment = timeline.environment.card, timeline.environment
        environment_parts = [_resolved(card.location, environment.location_override),
            _resolved(card.default_time_weather, environment.time_weather_override),
            _resolved(card.default_background, environment.background_override),
            _resolved(card.default_atmosphere, environment.atmosphere_override)]
        if environment_number:
            environment_parts.insert(0, _sentence(f"Use <Subject {environment_number}> as the environment reference. {card.reference.usage}"))

        continuous_events, timed_events, soundscape, music = [], [], [], []
        for track_index, track in enumerate(timeline.tracks.tracks):
            for clip_index, clip in enumerate(track.clips):
                rendered = _render_clip(track, clip, labels)
                event = (clip.start_time, clip.end_time, track_index, clip_index, track, rendered)
                if track.owner_kind != "actor" and clip.start_time <= 1e-6 and clip.end_time >= timeline.duration - 1e-6:
                    continuous_events.append(event)
                else:
                    timed_events.append(event)
                if clip.kind == "audio":
                    timed = f"From {_time(clip.start_time)} to {_time(clip.end_time)} seconds, {_sentence(clip.content)}"
                    (music if clip.audio_type == "music" else soundscape).append(timed)
        continuous_events.sort(key=lambda item: (item[2], item[3]))
        timed_events.sort(key=lambda item: (item[0], item[2], item[3]))
        timeline_lines = []
        actor_steps = {}
        actor_end_times = {}
        for start_time, end_time, _, _, track, rendered in timed_events:
            if track.owner_kind == "actor":
                owner = id(track.owner)
                step = actor_steps.get(owner, 0)
                if step == 0:
                    marker = f"For {labels[owner]}, first"
                elif start_time >= actor_end_times[owner] - 1e-6:
                    marker = f"For {labels[owner]}, then, only after the preceding action has ended"
                else:
                    marker = f"For {labels[owner]}, meanwhile at {_time(start_time)} seconds"
                actor_steps[owner] = step + 1
                actor_end_times[owner] = max(end_time, actor_end_times.get(owner, end_time))
            else:
                marker = f"At {_time(start_time)} seconds"
            timeline_lines.append(f"{marker}: {rendered}")
        order_rule = (
            "The timeline order is mandatory. Do not anticipate, swap, or perform any later action before its stated start time. "
            "Actions marked 'then' begin only after the preceding action has completely ended."
        )
        detailed = " ".join(filter(_text, ["[Shot 1]", *style_parts, *environment_parts, *character_lines,
            "Strict chronological timeline:", order_rule, *timeline_lines,
            "Continuous non-character controls:", *(item[5] for item in continuous_events),
            _sentence(additional_instructions)]))

        if prompt_format == "Ref":
            numbers = sorted(definitions)
            subjects = [f"<Subject {number}>" for number in numbers]
            subject_text = subjects[0] if len(subjects) == 1 else " and ".join(subjects) if len(subjects) == 2 else ", ".join(subjects[:-1]) + f", and {subjects[-1]}"
            result = ["subject_definitions:\n" + "\n".join(definitions[number] for number in numbers),
                "summary:\n" + f"[reference generation] The target video is a {settings.duration:.2f}-second continuous single shot, using {subject_text} as the referenced visible content.",
                "retention_analysis:\n" + "\n".join(f"<Subject {number}> (appears in [Shot 1]): {retentions[number][0]} - {retentions[number][1] or 'Preserve only the defined reference role.'}" for number in numbers),
                "detailed_description:\n" + detailed, "overall_soundscape:\n" + (" ".join(soundscape) or "N/A"),
                "non_diegetic_music:\n" + (" ".join(music) or "N/A")]
        else:
            result = ["integrated_multimodal_description: " + detailed,
                "overall_soundscape: " + (" ".join(soundscape) or "N/A"),
                "non_diegetic_music: " + (" ".join(music) or "N/A")]
        return io.NodeOutput(CompletePromptData(_bind_actor_tokens("\n\n".join(result), actor_labels), tuple(references), settings))


class MiniMaxH3PromptParser(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PromptParser", display_name="MiniMax H3 提示词解析（Prompt Parser）", category=CATEGORY,
            inputs=[H3_PROMPT.Input("complete_prompt")], outputs=[io.String.Output(display_name="prompt"),
            io.Image.Output(display_name="first_frame"), io.Image.Output(display_name="last_frame"),
            io.Int.Output(display_name="width"), io.Int.Output(display_name="height"), io.Int.Output(display_name="length"),
            io.Image.Output(display_name="reference_images", is_output_list=True), io.Int.Output(display_name="picture_numbers", is_output_list=True)])

    @classmethod
    def execute(cls, complete_prompt):
        settings = complete_prompt.video_settings
        return io.NodeOutput(complete_prompt.text, settings.first_frame, settings.last_frame, settings.width, settings.height,
            settings.length, [item.image for item in complete_prompt.references], [item.picture_number for item in complete_prompt.references])


class MiniMaxH3Ref2VAAdapter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Ref2VAAdapter", display_name="MiniMax H3 原生参考生视频适配（Ref2VA Adapter）",
            category=CATEGORY, is_input_list=True, inputs=[io.Clip.Input("clip"), io.Vae.Input("vae"), io.Vae.Input("audio_vae"),
            io.String.Input("prompt", display_name="提示词", placeholder="提示词", multiline=True), io.Int.Input("width"), io.Int.Input("height"), io.Int.Input("length"),
            io.Image.Input("reference_images"), io.Int.Input("picture_numbers"),
            io.Combo.Input("ref_image_size", options=["match", "max"], default="match")],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()])

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, reference_images, picture_numbers, ref_image_size):
        if picture_numbers != list(range(1, len(reference_images) + 1)):
            raise ValueError(f"Reference images require consecutive picture numbers; got {picture_numbers}")
        return NativeRef2VA.execute(clip=clip[0], vae=vae[0], audio_vae=audio_vae[0], prompt=prompt[0], width=width[0],
            height=height[0], length=length[0], ref_image_size=ref_image_size[0],
            ref_images={f"ref_image_{index}": image for index, image in enumerate(reference_images)})


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
            if reference is None:
                continue
            owner = labels.get(id(track.owner), "the character")
            line = (f"For {owner}'s action from {_time(clip.start_time)} to {_time(clip.end_time)} seconds, "
                    f"use <Video {video_number}> as the motion reference. {role_text[reference.role]}")
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
    leading_seconds = generation_job.overlap_seconds if has_previous_segment else 0.0
    timeline = _segment_timeline(generation_job.timeline, start, end, leading_seconds)
    state = _persistent_state(generation_job.timeline, start)
    first_video_number = 2 if has_previous_segment else 1
    motion_references, motion_instructions = _motion_references(timeline, first_video_number)
    continuity_instruction = ""
    if has_previous_segment:
        continuity_instruction = _sentence(
            f"The opening {_time(leading_seconds)} seconds must continue the end of <Video 1>: preserve the same subjects, "
            "poses, spatial arrangement, movement direction, velocity, environment, lighting, and camera state. "
            "After this overlap, perform the current segment's actions in their stated order. "
            "Do not replay earlier dialogue or completed actions"
        )
    total_videos = len(motion_references) + (1 if has_previous_segment else 0)
    if total_videos > 3:
        raise ValueError("单个生成片段最多支持 3 段参考视频；后续片段需为连续性视频保留 1 个位置，因此最多连接 2 个动作参考视频")
    additional = " ".join(filter(_text, (state, continuity_instruction, motion_instructions)))
    compiled = MiniMaxH3FinalPrompt.execute(timeline, generation_job.megapixels, generation_job.aspect_ratio,
        prompt_format="Ref", additional_instructions=additional)[0]
    return compiled, motion_references


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
            io.Float.Input("continuity_seconds", display_name="连续性参考长度（秒）", default=2.0, min=0.25, max=15.0, step=0.25),
            io.Float.Input("overlap_seconds", display_name="重叠匹配长度（秒）", default=0.5, min=0.1, max=2.0, step=0.05)],
            outputs=[H3_GENERATION_JOB.Output(display_name="generation_job")])

    @classmethod
    def execute(cls, timeline, megapixels, aspect_ratio, seed, scheduler, steps, denoise, ref_image_size,
                continuity_seconds, overlap_seconds):
        if not isinstance(timeline, TimelineData):
            raise TypeError("生成任务包需要 MiniMax H3 时间轴")
        _validate_timeline(timeline)
        _segment_ranges(timeline)
        return io.NodeOutput(GenerationJobData(timeline, megapixels, aspect_ratio, seed, scheduler, steps, denoise,
            ref_image_size, continuity_seconds, overlap_seconds))


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
        video_index = 0
        if previous_tail_video is not None:
            ref_videos["ref_video_0"] = (_match_reference_video(previous_tail_video, settings.width, settings.height)
                if generation_job.ref_image_size == "match" else previous_tail_video)
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
            ref_images={f"ref_image_{index}": item.image for index, item in enumerate(compiled.references)},
            ref_videos=ref_videos, ref_video_audios=ref_video_audios)


class MiniMaxH3PromptPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PromptPreview", display_name="MiniMax H3 最终提示词预览（Final Prompt Preview）",
            category=CATEGORY, description="按实际多段 Ref2VA 生成顺序预览每个片段的最终完整提示词。",
            inputs=[H3_GENERATION_JOB.Input("generation_job")], outputs=[io.String.Output(display_name="final_prompts")],
            is_output_node=True)

    @classmethod
    def execute(cls, generation_job):
        if not isinstance(generation_job, GenerationJobData):
            raise TypeError("最终提示词预览需要 MiniMax H3 生成任务包")
        ranges = _segment_ranges(generation_job.timeline)
        sections = []
        for index, (start, end) in enumerate(ranges):
            has_previous = index > 0
            compiled, motion_references = _compile_generation_segment(generation_job, index, has_previous)
            references = []
            if has_previous:
                references.append("<Video 1> = 上一片段尾部连续性视频")
            first_motion_number = 2 if has_previous else 1
            references.extend(f"<Video {first_motion_number + offset}> = 当前片段动作参考视频 {offset + 1}"
                for offset in range(len(motion_references)))
            generated_duration = end - start + (generation_job.overlap_seconds if has_previous else 0.0)
            header = [f"========== 片段 {index + 1}/{len(ranges)} ==========",
                f"原时间轴范围：{_time(start)}–{_time(end)} 秒",
                f"本次生成时长：{_time(generated_duration)} 秒",
                *(references or ["参考视频：无"]), "", compiled.text]
            sections.append("\n".join(header))
        preview = "\n\n".join(sections)
        return io.NodeOutput(preview, ui={"text": (preview,)})


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
        return io.Schema(node_id="MiniMaxH3SegmentJoin", display_name="MiniMax H3 分段拼接（内部）",
            category=f"{CATEGORY}/内部", inputs=[io.Image.Input("previous_images"), io.Audio.Input("previous_audio"),
            io.Image.Input("current_images"), io.Audio.Input("current_audio"),
            io.Float.Input("overlap_seconds", default=0.5, min=0.1, max=2.0, step=0.05)],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio")])

    @classmethod
    def execute(cls, previous_images, previous_audio, current_images, current_audio, overlap_seconds):
        if previous_images.shape[1:] != current_images.shape[1:]:
            raise ValueError("分段画面尺寸不一致，无法拼接")
        if previous_audio["sample_rate"] != current_audio["sample_rate"]:
            raise ValueError("分段音频采样率不一致，无法拼接")
        search_count = min(current_images.shape[0], max(1, round(overlap_seconds * FPS)))
        previous = F.interpolate(previous_images[-2:].movedim(-1, 1).float(), size=(32, 32),
            mode="bilinear", align_corners=False)
        current = F.interpolate(current_images[:search_count].movedim(-1, 1).float(), size=(32, 32),
            mode="bilinear", align_corners=False)
        previous_gray = previous.mean(dim=1)
        current_gray = current.mean(dim=1)
        previous_structure = ((previous_gray - previous_gray.mean(dim=(1, 2), keepdim=True))
            / previous_gray.std(dim=(1, 2), keepdim=True).clamp_min(1e-5))
        current_structure = ((current_gray - current_gray.mean(dim=(1, 2), keepdim=True))
            / current_gray.std(dim=(1, 2), keepdim=True).clamp_min(1e-5))
        scores = 0.65 * ((current - previous[-1:]) ** 2).mean(dim=(1, 2, 3))
        scores += 0.35 * ((current_structure - previous_structure[-1:]) ** 2).mean(dim=(1, 2))
        if previous.shape[0] == 2 and current.shape[0] > 1:
            previous_motion = previous[-1] - previous[-2]
            current_motion = current[1:] - current[:-1]
            scores[1:] += 0.25 * ((current_motion - previous_motion) ** 2).mean(dim=(1, 2, 3))
        target = search_count - 1
        if target > 0:
            timing = (torch.arange(search_count, device=scores.device) - target).float() / target
            scores += 0.02 * timing.square()
        skip_frames = int(scores.argmin().item()) + 1
        sample_rate = current_audio["sample_rate"]
        skip_samples = min(current_audio["waveform"].shape[-1], round(skip_frames * sample_rate / FPS))
        images = torch.cat((previous_images, current_images[skip_frames:]), dim=0)
        waveform = torch.cat((previous_audio["waveform"], current_audio["waveform"][..., skip_samples:]), dim=-1)
        return io.NodeOutput(images, {**current_audio, "waveform": waveform})


class MiniMaxH3MultiSegmentGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3MultiSegmentGenerate", display_name="MiniMax H3 多段生成（Decoded Video）",
            category=CATEGORY, description="自动逐段生成、解码并拼接；节点内显示当前片段、阶段和总体进度。",
            inputs=[io.Model.Input("model"), io.Clip.Input("clip"), io.Vae.Input("video_vae"),
            io.Vae.Input("audio_vae"), io.Sampler.Input("sampler"), H3_GENERATION_JOB.Input("generation_job")],
            outputs=[io.Video.Output(display_name="video")], enable_expand=True)

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
            conditioning_inputs = {"clip": clip, "video_vae": video_vae, "audio_vae": audio_vae,
                "generation_job": generation_job, "segment_index": index}
            if previous_tail_video is not None:
                conditioning_inputs["previous_tail_video"] = previous_tail_video
            conditioning = stage_node("MiniMaxH3SegmentConditioning", f"{stage}_conditioning", **conditioning_inputs)
            noise = stage_node("RandomNoise", f"{stage}_noise",
                noise_seed=(generation_job.seed + index) & 0xffffffffffffffff)
            guider = stage_node("BasicGuider", f"{stage}_guider", model=model, conditioning=conditioning.out(0))
            sigmas = stage_node("BasicScheduler", f"{stage}_scheduler", model=model, scheduler=generation_job.scheduler,
                steps=generation_job.steps, denoise=generation_job.denoise)
            sampled = stage_node("SamplerCustomAdvanced", f"{stage}_sampling", noise=noise.out(0),
                guider=guider.out(0), sampler=sampler, sigmas=sigmas.out(0), latent_image=conditioning.out(1))
            images = stage_node("VAEDecode", f"{stage}_video_decode", samples=sampled.out(0), vae=video_vae).out(0)
            audio = stage_node("VAEDecodeAudio", f"{stage}_audio_decode", samples=sampled.out(0), vae=audio_vae).out(0)
            segment_duration = ranges[index][1] - ranges[index][0]
            if index > 0:
                segment_duration += generation_job.overlap_seconds
            trimmed = stage_node("MiniMaxH3SegmentTrim", f"{stage}_trim", images=images, audio=audio,
                duration_seconds=segment_duration)
            images = trimmed.out(0)
            audio = trimmed.out(1)
            if index < len(ranges) - 1:
                previous_tail_video = stage_node("MiniMaxH3ContinuityTail", f"{stage}_continuity", images=images,
                    continuity_seconds=generation_job.continuity_seconds).out(0)
            if accumulated_images is None:
                accumulated_images = images
                accumulated_audio = audio
            else:
                joined = stage_node("MiniMaxH3SegmentJoin", f"{stage}_join", previous_images=accumulated_images,
                    previous_audio=accumulated_audio, current_images=images, current_audio=audio,
                    overlap_seconds=generation_job.overlap_seconds)
                accumulated_images = joined.out(0)
                accumulated_audio = joined.out(1)
        video = stage_node("CreateVideo", f"segment_{len(ranges)}_of_{len(ranges)}_final_video",
            images=accumulated_images, fps=float(FPS), audio=accumulated_audio)
        return io.NodeOutput(video.out(0), expand=graph.finalize())


class MiniMaxH3PromptBuilderExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [MiniMaxH3Character, MiniMaxH3ActorInstance, MiniMaxH3CharacterGroup, MiniMaxH3Language, MiniMaxH3Visual,
            MiniMaxH3Environment, MiniMaxH3EnvironmentInstance, MiniMaxH3Action, MiniMaxH3Camera, MiniMaxH3LightingAction,
            MiniMaxH3AudioAction, MiniMaxH3EnvironmentAction, MiniMaxH3ActorTrack, MiniMaxH3EnvironmentTrack,
            MiniMaxH3SystemTrack, MiniMaxH3TrackList, MiniMaxH3Timeline, MiniMaxH3FinalPrompt,
            MiniMaxH3PromptParser, MiniMaxH3Ref2VAAdapter, MiniMaxH3GenerationJob, MiniMaxH3MotionReference,
            MiniMaxH3PromptPreview, MiniMaxH3SegmentConditioning, MiniMaxH3ContinuityTail, MiniMaxH3SegmentTrim, MiniMaxH3SegmentJoin,
            MiniMaxH3MultiSegmentGenerate]


async def comfy_entrypoint():
    return MiniMaxH3PromptBuilderExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
