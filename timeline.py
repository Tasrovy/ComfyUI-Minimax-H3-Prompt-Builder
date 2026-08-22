from dataclasses import replace

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo as NativeRef2VA

from .schema import (ASPECT_RATIOS, CATEGORY, FPS, H3_CHARACTER_GROUP, H3_ENVIRONMENT_INSTANCE,
    H3_PROMPT, H3_STYLE_CARD, H3_TIMELINE, H3_TRACK_LIST, CompletePromptData,
    ReferenceImageData, TimelineData, TrackListData, VideoSettingsData)
from .utils import (_actor_state, _bind_actor_tokens, _card_description, _clip_has_content,
    _lower_first, _resolved, _same_image, _sentence, _text, _time, _video_length, _video_size)


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
        tracks = tuple(replace(track, clips=tuple(clip for clip in track.clips if _clip_has_content(clip)))
            for track in tracks.tracks)
        return io.NodeOutput(TimelineData(character_group, style_card, environment, TrackListData(tracks), duration_seconds))


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


def _render_clip(track, clip, labels, timeline_duration):
    content = _text(clip.content)
    quality = clip.quality
    end_state = clip.result
    parts = []
    if track.owner_kind == "actor":
        label = labels[id(track.owner)]
        end_state = _actor_state(label, clip.result, "")
        if content:
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
            parts.append(_sentence(text))
    elif content:
        if track.owner_kind == "environment":
            text = f"the environment changes: {content}"
        elif clip.kind == "lighting":
            text = f"the lighting changes: {content}"
        elif clip.kind == "audio":
            text = f"the {clip.audio_type} sound is heard: {content}"
        else:
            text = content
        parts.append(_sentence(text))
    if quality:
        parts.append(quality)
    if end_state:
        parts.append(_sentence(f"Afterward, {end_state}"))
    if not parts:
        return ""
    covers_shot = clip.start_time <= 1e-6 and clip.end_time >= timeline_duration - 1e-6
    prefix = "" if covers_shot or not content else f"From {_time(clip.start_time)} to {_time(clip.end_time)} seconds, "
    return prefix + " ".join(parts)


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
    def execute(cls, timeline, megapixels, aspect_ratio, prompt_format="Ref", first_frame=None, last_frame=None,
                additional_instructions="", motion_instructions="", empty_sections="不输出", continuity_keyframe=False,
                suppress_initial_state=False, generation_duration=None, motion_definitions="",
                motion_retentions="", motion_summary="", suppress_actor_state_ids=(), suppress_camera_tracks=False):
        if not isinstance(timeline, TimelineData):
            raise TypeError("Timeline compiler requires TimelineData")
        _validate_timeline(timeline)
        if prompt_format == "Ref" and (first_frame is not None or last_frame is not None):
            raise ValueError("Ref mode does not use first_frame or last_frame")
        width, height = _video_size(megapixels, aspect_ratio)
        length = _video_length(generation_duration if generation_duration is not None else timeline.duration)
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
            description = _card_description(card.description, card.name)
            number = None
            if prompt_format == "Ref":
                number = add_reference(card.reference,
                    f"<Subject {{number}}> is {card.name}, whose identity and appearance come from <Picture {{number}}>. {card.name} is {_lower_first(description)}",
                    "fully_preserved", card.preservation or
                    "Preserve identity and appearance while allowing new pose and framing.")
            identity = f"{label} is <Subject {number}>" if number else (f"{label} is {_lower_first(description)}" if description else "")
            if identity:
                character_lines.append(_sentence(identity))
                
            if not suppress_initial_state:
                use_defaults = id(actor) not in suppress_actor_state_ids
                character_lines.extend(filter(_text, (_actor_state(label, card.default_position if use_defaults else "", actor.position_override),
                    _actor_state(label, card.default_pose if use_defaults else "", actor.pose_override),
                    _actor_state(label, card.default_emotion if use_defaults else "", actor.emotion_override),
                    _actor_state(label, card.default_appearance, actor.appearance_override))))
                    
            if card.character_style:
                rule = "prioritize this character-specific style over conflicting global style" if card.style_priority == "character" else "the global style takes priority; use this character style only where compatible"
                character_lines.append(_sentence(f"For {label}, {rule}: {_card_description(card.character_style, card.name)}"))

        style_number = None
        environment_number = None
        if prompt_format == "Ref":
            style_number = add_reference(timeline.style.reference,
                "<Subject {number}> is the visual style derived from <Picture {number}>.", "partially_preserved",
                (timeline.style.reference.usage or "Use its visual style without preserving its subjects or composition.")
                if timeline.style.reference else "")
            environment_number = add_reference(timeline.environment.card.reference,
                f"<Subject {{number}}> is the {timeline.environment.card.name} environment derived from <Picture {{number}}>.",
                "partially_preserved", timeline.environment.card.preservation or
                "Preserve the recognizable environment while allowing new framing and subject placement.")
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
            if suppress_camera_tracks and track.owner_kind == "camera":
                continue
            for clip_index, clip in enumerate(track.clips):
                if clip.kind == "audio" and _text(clip.content):
                    covers_shot = clip.start_time <= 1e-6 and clip.end_time >= timeline.duration - 1e-6
                    timed = (_sentence(clip.content) if covers_shot else
                        f"From {_time(clip.start_time)} to {_time(clip.end_time)} seconds, {_sentence(clip.content)}")
                    (music if clip.audio_type == "music" else soundscape).append(timed)
                    continue
                rendered = _render_clip(track, clip, labels, timeline.duration)
                event = (clip.start_time, clip.end_time, track_index, clip_index, track, rendered)
                if track.owner_kind != "actor" and clip.start_time <= 1e-6 and clip.end_time >= timeline.duration - 1e-6:
                    continuous_events.append(event)
                else:
                    timed_events.append(event)
        continuous_events.sort(key=lambda item: (item[2], item[3]))
        owner_priority = {"actor": 0, "environment": 1, "camera": 2, "lighting": 3}
        timed_events.sort(key=lambda item: (item[0], owner_priority.get(item[4].owner_kind, 4), item[2], item[3]))
        timeline_lines = []
        actor_end_times = {}
        for start_time, end_time, _, _, track, rendered in timed_events:
            if not _text(rendered):
                continue
            if track.owner_kind == "actor" and id(track.owner) in actor_end_times:
                relation = "Then, " if start_time >= actor_end_times[id(track.owner)] - 1e-6 else "Meanwhile, "
                rendered = relation + _lower_first(rendered)
            if track.owner_kind == "actor":
                actor_end_times[id(track.owner)] = max(end_time, actor_end_times.get(id(track.owner), end_time))
            timeline_lines.append(rendered)
        continuity_number = len(references) + 1 if continuity_keyframe else None
        continuity_parts = []
        if continuity_keyframe:
            continuity_parts.extend([_sentence(
                f"How the reference pictures align with the target video - <Picture {continuity_number}> aligns with "
                "the 0.00-second mark of the target video as the exact opening frame."),
                _sentence(f"At 0.00 seconds the frame is exactly <Picture {continuity_number}> without reinterpretation.")])
        if empty_sections == "输出 N/A":
            detailed_parts = [*continuity_parts, "[Shot 1]", *style_parts, *environment_parts, *character_lines,
                *timeline_lines]
            if continuous_events:
                detailed_parts.extend(item[5] for item in continuous_events)
            if additional_instructions:
                detailed_parts.append(_sentence(additional_instructions))
            if motion_instructions:
                detailed_parts.extend(["Motion references:", _sentence(motion_instructions)])
            detailed = " ".join(filter(_text, detailed_parts))
        else:
            body_parts = [*continuity_parts, *style_parts, *environment_parts, *character_lines]
            if timeline_lines:
                body_parts.extend(timeline_lines)
            if continuous_events:
                body_parts.extend(item[5] for item in continuous_events)
            if additional_instructions:
                body_parts.append(_sentence(additional_instructions))
            if motion_instructions:
                body_parts.extend(["Motion references:", _sentence(motion_instructions)])
            body = " ".join(filter(_text, body_parts))
            detailed = " ".join(filter(_text, ["[Shot 1]", body])) if body else ""

        if prompt_format == "Ref":
            numbers = sorted(definitions)
            subjects = [f"<Subject {number}>" for number in numbers]
            subject_text = subjects[0] if len(subjects) == 1 else " and ".join(subjects) if len(subjects) == 2 else ", ".join(subjects[:-1]) + f", and {subjects[-1]}"
            definitions_text = "\n".join(definitions[number] for number in numbers)
            retentions_text = "\n".join(f"<Subject {number}> (appears in [Shot 1]): {retentions[number][0]} - {retentions[number][1] or 'Preserve only the defined reference role.'}" for number in numbers)
            if motion_definitions:
                definitions_text += "\n" + motion_definitions
            if motion_retentions:
                retentions_text += "\n" + motion_retentions
            if continuity_keyframe:
                definitions_text += (f"\n<Picture {continuity_number}> is the exact opening frame of this segment at 0.00 seconds, "
                    f"extracted from the last frame of the previous segment without reinterpretation. "
                    f"<Video 1> is the previous segment's tail and provides motion continuity.")
                retentions_text += (f"\n<Picture {continuity_number}> (at 0.00s): fully_preserved - exact opening frame; "
                    f"the first frame of this segment must match it. "
                    f"<Video 1>: partially_preserved - motion continuity reference; continue the preceding movement; do not restart it.")
            summary_tag = "[keyframe completion + reference generation]" if continuity_keyframe else "[reference generation]"
            primary = next(((track, clip) for track in timeline.tracks.tracks if track.owner_kind == "actor"
                for clip in track.clips if clip.kind == "body" and _text(clip.content)), None)
            if primary is None:
                primary = next(((track, clip) for track in timeline.tracks.tracks if track.owner_kind == "actor"
                    for clip in track.clips if _text(clip.content)), None)
            summary = (f"{summary_tag} Generate a continuous single shot." if generation_duration is not None else
                f"{summary_tag} The target video is a {_time(timeline.duration)}-second continuous single shot.")
            if primary is not None:
                track, clip = primary
                summary += f" Main visible action: {labels[id(track.owner)]} {_lower_first(_text(clip.content))}."
            summary += f" It uses {subject_text} as the referenced content."
            if motion_summary:
                summary += " " + motion_summary
            result = ["subject_definitions:\n" + definitions_text,
                "summary:\n" + summary,
                "retention_analysis:\n" + retentions_text]
            if empty_sections == "输出 N/A":
                result.extend(["detailed_description:\n" + detailed,
                    "overall_soundscape:\n" + (" ".join(soundscape) or "N/A"),
                    "non_diegetic_music:\n" + (" ".join(music) or "N/A")])
            else:
                if detailed:
                    result.append("detailed_description:\n" + detailed)
                if soundscape:
                    result.append("overall_soundscape:\n" + " ".join(soundscape))
                if music:
                    result.append("non_diegetic_music:\n" + " ".join(music))
        else:
            result = []
            if empty_sections == "输出 N/A":
                result.extend(["integrated_multimodal_description: " + detailed,
                    "overall_soundscape: " + (" ".join(soundscape) or "N/A"),
                    "non_diegetic_music: " + (" ".join(music) or "N/A")])
            else:
                if detailed:
                    result.append("integrated_multimodal_description: " + detailed)
                if soundscape:
                    result.append("overall_soundscape: " + " ".join(soundscape))
                if music:
                    result.append("non_diegetic_music: " + " ".join(music))
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
