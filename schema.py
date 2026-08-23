from dataclasses import dataclass

from comfy_api.latest import io


CATEGORY = "MiniMax H3/提示词构建"
FPS = 24
ASPECT_RATIOS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "21:9": (21, 9)}
ACTOR_KINDS = ("body", "expression", "gaze", "speech")
SYSTEM_KINDS = ("camera", "lighting", "audio")
EMPTY_SECTION_MODES = ("自动补全", "输出 N/A")
EMPTY_SECTION_OPTIONS = list(EMPTY_SECTION_MODES) + ["不输出", 1, 2]

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
    source_duration: float = 0.0
    aligned_duration: float = 0.0
    motion_duration: float = 0.0
    context_duration: float = 0.0


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
    rendered_video: object | None = None
    rendered_video_version: int = 0


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
    empty_sections: str


for custom_type, data_type in ((H3_CHARACTER_CARD, CharacterCardData), (H3_ACTOR_INSTANCE, ActorInstanceData),
                               (H3_CHARACTER_GROUP, CharacterGroupData), (H3_LANGUAGE, LanguageData),
                               (H3_STYLE_CARD, StyleCardData), (H3_ENVIRONMENT_CARD, EnvironmentCardData),
                               (H3_ENVIRONMENT_INSTANCE, EnvironmentInstanceData), (H3_MOTION_REFERENCE, MotionReferenceData),
                               (H3_TIMELINE_CLIP, TimelineClipData),
                               (H3_TIMELINE_TRACK, TimelineTrackData), (H3_TRACK_LIST, TrackListData),
                               (H3_TIMELINE, TimelineData), (H3_PROMPT, CompletePromptData),
                               (H3_GENERATION_JOB, GenerationJobData)):
    custom_type.Type = data_type
