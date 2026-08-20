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


