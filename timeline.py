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


def _speaker_ids(timeline):
    events = sorted(((clip.start_time, track_index, clip_index, id(track.owner))
        for track_index, track in enumerate(timeline.tracks.tracks) if track.owner_kind == "actor"
        for clip_index, clip in enumerate(track.clips) if clip.kind == "speech" and _text(clip.content)),
        key=lambda item: (item[0], item[1], item[2]))
    speakers = {}
    for _, _, _, actor_id in events:
        if actor_id not in speakers:
            speakers[actor_id] = f"S{len(speakers) + 1}"
    return speakers


def _reference_subject_layout(timeline):
    actor_subjects = {}
    next_number = 1
    for actor in timeline.characters.actors:
        if actor.card.reference is not None:
            actor_subjects[id(actor)] = next_number
            next_number += 1
    style_subject = next_number if timeline.style.reference is not None else None
    next_number += style_subject is not None
    environment_subject = next_number if timeline.environment.card.reference is not None else None
    next_number += environment_subject is not None
    return actor_subjects, style_subject, environment_subject, next_number - 1


def _render_clip(track, clip, labels, subject_labels, speaker_ids, timeline_duration, time_offset=0.0):
    content = _text(clip.content)
    quality = clip.quality
    end_state = clip.result
    if ((track.owner_kind == "camera" and clip.camera_reference is not None) or
            (track.owner_kind == "lighting" and clip.lighting_reference is not None)):
        content = ""
        quality = ""
        end_state = ""
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
                speaker = f"{subject_labels.get(id(track.owner), label)} ({speaker_ids[id(track.owner)]})"
                text = f"{speaker}{mode}{voice}{delivery}: <d>[{clip.language.language}] {content}</d>"
                if clip.speech_type == "off-screen voiceover":
                    text += f" while {label}'s lips remain completely closed"
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
    start_time = clip.start_time + time_offset
    end_time = clip.end_time + time_offset
    covers_shot = start_time <= 1e-6 and end_time >= timeline_duration - 1e-6
    prefix = "" if covers_shot or not content else f"From {_time(start_time)} to {_time(end_time)} seconds, "
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
                additional_instructions="", motion_instructions=None, empty_sections="自动补全",
                suppress_initial_state=False, generation_duration=None, motion_definitions="",
                motion_retentions="", motion_summary="", suppress_actor_state_ids=(), suppress_camera_tracks=False,
                timeline_offset=0.0, opening_instructions="", continuation=False):
        if not isinstance(timeline, TimelineData):
            raise TypeError("Timeline compiler requires TimelineData")
        _validate_timeline(timeline)
        if prompt_format == "Ref" and (first_frame is not None or last_frame is not None):
            raise ValueError("Ref mode does not use first_frame or last_frame")
        width, height = _video_size(megapixels, aspect_ratio)
        length = _video_length(generation_duration if generation_duration is not None else timeline.duration)
        actual_duration = length / FPS
        if prompt_format == "Ref":
            mode = "Ref2VA"
        elif first_frame is not None and last_frame is not None:
            mode = "FL2VA"
        elif first_frame is not None:
            mode = "I2VA"
        elif last_frame is not None:
            mode = "L2VA"
        else:
            mode = "T2VA"
        settings = VideoSettingsData(mode, width, height, length, actual_duration, first_frame, last_frame)
        references = []

        def add_picture(reference):
            if reference is None:
                return None
            existing = next((item for item in references if _same_image(item.image, reference.image)), None)
            if existing is None:
                number = len(references) + 1
                references.append(ReferenceImageData(number, reference.image, reference.role, reference.usage))
            else:
                number = existing.picture_number
            return number

        actor_subjects, style_subject, environment_subject, _ = _reference_subject_layout(timeline)
        definitions = {}
        retentions = {}
        actor_labels = tuple(actor.card.name or f"Actor {index}" for index, actor in enumerate(timeline.characters.actors, 1))
        labels = {id(actor): actor_labels[index] for index, actor in enumerate(timeline.characters.actors)}
        subject_labels = {actor_id: f"<Subject {number}>" for actor_id, number in actor_subjects.items()}
        speaker_ids = _speaker_ids(timeline)
        active_actor_ids = {id(track.owner) for track in timeline.tracks.tracks
            if track.owner_kind == "actor" and any(_clip_has_content(clip) for clip in track.clips)}
        suppressed_actor_ids = active_actor_ids | set(suppress_actor_state_ids)
        character_lines = []
        for actor in timeline.characters.actors:
            card, label = actor.card, labels[id(actor)]
            description = _card_description(card.description, card.name)
            subject_number = actor_subjects.get(id(actor))
            if prompt_format == "Ref":
                picture_number = add_picture(card.reference)
                if subject_number is not None:
                    definition = f"<Subject {subject_number}> is {label}, whose identity and appearance come from <Picture {picture_number}>."
                    if description:
                        definition += f" {label} is {_lower_first(description)}"
                    definitions[subject_number] = definition
                    retentions[subject_number] = ("fully_preserved", card.preservation or
                        "Preserve identity and appearance while allowing new pose and framing.")
            identity = (f"{label} is represented by <Subject {subject_number}>" if subject_number is not None else
                (f"{label} is {_lower_first(description)}" if description else ""))
            if identity:
                character_lines.append(_sentence(identity))
                
            if not suppress_initial_state and id(actor) not in suppressed_actor_ids:
                character_lines.extend(filter(_text, (_actor_state(label, card.default_position, actor.position_override),
                    _actor_state(label, card.default_pose, actor.pose_override),
                    _actor_state(label, card.default_emotion, actor.emotion_override),
                    _actor_state(label, card.default_appearance, actor.appearance_override))))
                    
            if card.character_style:
                rule = "prioritize this character-specific style over conflicting global style" if card.style_priority == "character" else "the global style takes priority; use this character style only where compatible"
                character_lines.append(_sentence(f"For {label}, {rule}: {_card_description(card.character_style, card.name)}"))

        if prompt_format == "Ref":
            if style_subject is not None:
                picture_number = add_picture(timeline.style.reference)
                definitions[style_subject] = f"<Subject {style_subject}> is the visual style derived from <Picture {picture_number}>."
                retentions[style_subject] = ("partially_preserved", timeline.style.reference.usage or
                    "Use its visual style without preserving its subjects or composition.")
            if environment_subject is not None:
                picture_number = add_picture(timeline.environment.card.reference)
                definitions[environment_subject] = (f"<Subject {environment_subject}> is the "
                    f"{timeline.environment.card.name} environment derived from <Picture {picture_number}>.")
                retentions[environment_subject] = ("partially_preserved", timeline.environment.card.preservation or
                    "Preserve the recognizable environment while allowing new framing and subject placement.")
        if prompt_format == "Ref" and not references:
            raise ValueError("Ref mode requires at least one character, style, or environment reference image")

        style_parts = list(filter(_text, map(_sentence,
            (timeline.style.style, timeline.style.rendering, timeline.style.color_palette, timeline.style.texture))))
        if style_subject is not None:
            style_parts.append(_sentence(f"Use <Subject {style_subject}> as the global visual reference. {timeline.style.reference.usage}"))
        card, environment = timeline.environment.card, timeline.environment
        environment_parts = [_resolved(card.location, environment.location_override),
            _sentence(environment.time_weather_override),
            _resolved(card.default_background, environment.background_override),
            _sentence(environment.atmosphere_override)]
        if environment_subject is not None:
            environment_parts.insert(0, _sentence(
                f"Use <Subject {environment_subject}> as the environment reference. {card.reference.usage}"))

        opening_events, timed_events, soundscape, music = [], [], [], []
        motion_instructions = motion_instructions or {}
        for track_index, track in enumerate(timeline.tracks.tracks):
            if suppress_camera_tracks and track.owner_kind == "camera":
                continue
            for clip_index, clip in enumerate(track.clips):
                motion_text = motion_instructions.get(id(clip), "")
                if clip.kind == "audio" and (_text(clip.content) or motion_text):
                    covers_shot = (timeline_offset <= 1e-6 and clip.start_time <= 1e-6 and
                        clip.end_time >= timeline.duration - 1e-6)
                    audio_text = " ".join(filter(_text, (_sentence(clip.content), clip.quality, motion_text)))
                    if clip.audio_type == "music":
                        music.append(audio_text if covers_shot else
                            f"From {_time(clip.start_time + timeline_offset)} to {_time(clip.end_time + timeline_offset)} seconds, {audio_text}")
                        continue
                    if covers_shot:
                        soundscape.append(audio_text)
                        continue
                rendered = _render_clip(track, clip, labels, subject_labels, speaker_ids,
                    actual_duration, timeline_offset)
                if motion_text:
                    rendered = " ".join(filter(_text, (rendered, motion_text)))
                event = (clip.start_time + timeline_offset, clip.end_time + timeline_offset,
                    track_index, clip_index, track, rendered)
                if (timeline_offset <= 1e-6 and track.owner_kind != "actor" and
                        clip.start_time <= 1e-6 and clip.end_time >= timeline.duration - 1e-6):
                    opening_events.append(event)
                else:
                    timed_events.append(event)
        opening_priority = {"camera": 0, "environment": 1, "lighting": 2, "audio": 3}
        opening_events.sort(key=lambda item: (opening_priority.get(item[4].owner_kind, 4), item[2], item[3]))
        owner_priority = {"camera": 0, "environment": 1, "lighting": 2, "actor": 3, "audio": 4}
        timed_events.sort(key=lambda item: (item[0], owner_priority.get(item[4].owner_kind, 5), item[2], item[3]))
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
        frame_anchors = []
        if mode == "I2VA":
            frame_anchors.append("The shot begins from <Picture 1>, preserving its subjects, composition, lighting, and spatial relationships.")
        elif mode == "FL2VA":
            frame_anchors.extend(("The shot begins from Picture 1, preserving its opening composition and visible state.",
                "The action and camera path progressively converge to Picture 2 as the exact final composition."))
        elif mode == "L2VA":
            frame_anchors.append("The action and camera path progressively converge to <Picture 1> as the exact final composition.")
        body_parts = [*frame_anchors, *(item[5] for item in opening_events), *environment_parts, *character_lines]
        if additional_instructions:
            body_parts.append(_sentence(additional_instructions))
        if opening_instructions:
            body_parts.append(_sentence(opening_instructions))
        body_parts.extend(timeline_lines)
        described_end = timeline_offset + timeline.duration
        last_timed_end = max((item[1] for item in timed_events if _text(item[5])), default=described_end)
        if last_timed_end < described_end - 1e-6:
            body_parts.append(f"From {_time(last_timed_end)} to {_time(described_end)} seconds, the established subject states "
                "continue naturally without an unintended reset or a new action.")
        if actual_duration > described_end + 1e-6:
            body_parts.append(f"From {_time(described_end)} to {_time(actual_duration)} seconds, the final visible state, "
                "framing, lighting, environment, and synchronized sound remain stable without introducing a new action.")
        shot_body = " ".join(filter(_text, body_parts))
        if prompt_format == "Ref":
            style_text = " ".join(style_parts) or "The target video maintains a coherent visual style across the entire shot."
            detailed = f"{style_text}\n[Shot 1] {shot_body}".strip()
        else:
            detailed = " ".join(filter(_text, ("[Shot 1]", *style_parts, shot_body)))

        missing_soundscape = ("N/A" if empty_sections == "输出 N/A" else
            "Natural environmental ambience and physically synchronized movement sounds remain spatially coherent throughout the shot.")
        soundscape_text = " ".join(soundscape) or missing_soundscape
        music_text = " ".join(music) or "N/A"

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
            task_types = (["video continuation"] if continuation else []) + ["reference generation"]
            if "<Audio " in motion_definitions:
                task_types.append("audio reference")
            summary_tag = "[" + " + ".join(task_types) + "]"
            primary = next(((track, clip) for track in timeline.tracks.tracks if track.owner_kind == "actor"
                for clip in track.clips if clip.kind == "body" and _text(clip.content)), None)
            if primary is None:
                primary = next(((track, clip) for track in timeline.tracks.tracks if track.owner_kind == "actor"
                    for clip in track.clips if _text(clip.content)), None)
            summary = f"{summary_tag} The target video is a {_time(actual_duration)}-second continuous single shot."
            if primary is not None:
                track, clip = primary
                summary += f" Main visible action: {labels[id(track.owner)]} {_lower_first(_text(clip.content))}."
            summary += f" It uses {subject_text} as the referenced content."
            if motion_summary:
                summary += " " + motion_summary
            result = ["subject_definitions:\n" + definitions_text,
                "summary:\n" + summary,
                "retention_analysis:\n" + retentions_text,
                "detailed_description:\n" + detailed,
                "overall_soundscape:\n" + soundscape_text,
                "non_diegetic_music:\n" + music_text]
        else:
            instruction = ""
            if mode == "I2VA":
                instruction = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
            elif mode == "FL2VA":
                instruction = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with "
                    f"the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {actual_duration:.2f}-second mark of the target video.")
            elif mode == "L2VA":
                instruction = ("How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with "
                    f"the {actual_duration:.2f}-second mark of the target video.")
            result = ["integrated_multimodal_description: " + detailed,
                "overall_soundscape: " + soundscape_text,
                "non_diegetic_music: " + music_text]
            if instruction:
                result.insert(0, instruction)
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
