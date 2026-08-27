import re

from comfy_api.latest import io

from .schema import (CATEGORY, H3_ACTOR_INSTANCE, H3_CHARACTER_CARD, H3_CHARACTER_GROUP,
    H3_ENVIRONMENT_CARD, H3_ENVIRONMENT_INSTANCE, H3_LANGUAGE, H3_STYLE_CARD,
    ActorInstanceData, CharacterCardData, CharacterGroupData, EnvironmentCardData,
    EnvironmentInstanceData, LanguageData, StyleCardData)
from .utils import _autogrow, _reference, _sentence, _text, _values


class MiniMaxH3Character(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3Character", display_name="MiniMax H3 人物卡（Character Card）", category=CATEGORY, inputs=[
            io.String.Input("name", display_name="人物名称", placeholder="人物名称", default="the young woman"),
            io.String.Input("description", display_name="人物外观", placeholder="只描述固定长相、体型、发型、服装和饰品，不写动作、姿势、表情或场景", default="A young woman with long black hair wearing a dark red coat.", multiline=True),
            io.Combo.Input("style_priority", display_name="风格优先级", options=["character", "global"], default="global"),
            io.String.Input("character_style", display_name="人物风格", placeholder="人物风格", default="", multiline=True, optional=True),
            io.Image.Input("reference_image", optional=True),
        ], outputs=[H3_CHARACTER_CARD.Output(display_name="character_card")])

    @classmethod
    def execute(cls, name, description, character_style, style_priority, reference_image=None):
        return io.NodeOutput(CharacterCardData(_text(name) or "the character", _sentence(description),
            "Preserve identity and fixed appearance throughout the video.",
            _reference(reference_image, "character identity"), "", "", "", "",
            _sentence(character_style), style_priority))


class MiniMaxH3ActorInstance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3ActorInstance", display_name="MiniMax H3 人物实例（Actor Instance）", category=CATEGORY,
            description="声明 actor_1 形式的人物宏，并保存没有动作片段时使用的备用静态状态。其他节点推荐用 {actor_1} 引用人物；直接写人物名仍可编译但不利于复用。", inputs=[H3_CHARACTER_CARD.Input("character_card"),
            io.String.Input("actor_id", display_name="人物实例宏", placeholder="例如 actor_1", default="actor_1"),
            io.String.Input("position_override", display_name="无动作时位置", placeholder="仅描述静态位置，不写移动过程", default="", multiline=True, optional=True),
            io.String.Input("pose_override", display_name="无动作时姿态", placeholder="仅描述静态姿态，不写动作过程", default="", multiline=True, optional=True),
            io.String.Input("emotion_override", display_name="无动作时表情", placeholder="仅描述静态表情，不写表情变化", default="", multiline=True, optional=True),
            io.String.Input("appearance_override", display_name="无动作时附加状态", placeholder="仅描述当前持有物等静态状态", default="", multiline=True, optional=True)],
            outputs=[H3_ACTOR_INSTANCE.Output(display_name="actor_instance")])

    @classmethod
    def execute(cls, character_card, position_override="", pose_override="", emotion_override="", appearance_override="",
                actor_id="actor_1"):
        if not isinstance(character_card, CharacterCardData):
            raise TypeError("Actor instance requires a character card")
        actor_id = _text(actor_id)
        if not re.fullmatch(r"actor_[1-9][0-9]*", actor_id):
            raise ValueError("人物实例宏必须使用 actor_1、actor_2 这样的格式")
        return io.NodeOutput(ActorInstanceData(character_card,
            *map(_text, (position_override, pose_override, emotion_override, appearance_override)), actor_id))


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
        if len({actor.actor_id for actor in actors}) != len(actors):
            raise ValueError("人物组中的人物实例宏不能重复")
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
            io.String.Input("location", display_name="空间与地点外观", placeholder="只描述空间结构、建筑和固定材质，不写天气、灯光、人物或事件", default="A pedestrian bridge above a city street.", multiline=True),
            io.String.Input("default_background", display_name="固定背景与陈设", placeholder="只描述固定背景、陈设和空间关系，不写变化过程", default="Distant traffic lanes and neon signs are visible beyond the railing.", multiline=True),
            io.Image.Input("reference_image", optional=True)], outputs=[H3_ENVIRONMENT_CARD.Output(display_name="environment_card")])

    @classmethod
    def execute(cls, name, location, default_background, reference_image=None):
        return io.NodeOutput(EnvironmentCardData(_text(name) or "the environment", _sentence(location), "",
            _sentence(default_background), "", "Preserve the location and spatial layout.",
            _reference(reference_image, "environment", "Use the picture for location and spatial layout only.")))


class MiniMaxH3EnvironmentInstance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3EnvironmentInstance", display_name="MiniMax H3 环境实例（Environment Instance）", category=CATEGORY,
            description="描述当前环境的静态状态；环境变化仍由环境动作片段负责。", inputs=[H3_ENVIRONMENT_CARD.Input("environment_card"),
            io.String.Input("location_override", display_name="当前地点变体", placeholder="可选：当前实例与环境卡不同的静态空间特征", default="", multiline=True, optional=True),
            io.String.Input("time_weather_override", display_name="当前时间与天气", placeholder="只描述当前状态，不写天气变化过程", default="", multiline=True, optional=True),
            io.String.Input("background_override", display_name="当前背景变体", placeholder="可选：当前实例不同的固定背景或陈设", default="", multiline=True, optional=True),
            io.String.Input("atmosphere_override", display_name="当前环境氛围", placeholder="只描述当前氛围，不写变化过程", default="", multiline=True, optional=True)],
            outputs=[H3_ENVIRONMENT_INSTANCE.Output(display_name="environment_instance")])

    @classmethod
    def execute(cls, environment_card, location_override="", time_weather_override="", background_override="", atmosphere_override=""):
        if not isinstance(environment_card, EnvironmentCardData):
            raise TypeError("Environment instance requires an environment card")
        return io.NodeOutput(EnvironmentInstanceData(environment_card, *map(_text,
            (location_override, time_weather_override, background_override, atmosphere_override))))


