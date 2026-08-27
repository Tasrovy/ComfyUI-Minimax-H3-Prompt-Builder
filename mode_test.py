import importlib.util
import json
import pathlib
import sys
import tempfile
from fractions import Fraction

import torch


ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT.parents[1]))

name = "h3_mode_test"
spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)

s = mod.schema
u = mod.utils

library = mod.resource_library.normalize_library({
    "format": "minimax-h3-resource-library", "version": 1, "library_id": "test-library", "name": "测试资源库",
    "characters": [{"id": "character_luluka", "display_name": "Luluka", "tags": ["蓝发"],
        "reference_image": {"type": "input", "subfolder": "minimax_h3/resources", "filename": "luluka.png"},
        "card": {"name": "Luluka", "description": "A short girl with long blue hair.", "style_priority": "global"},
        "instance_defaults": {"pose_override": "stands naturally"}}],
    "environments": [{"id": "environment_bridge", "display_name": "雨夜人行桥",
        "card": {"name": "bridge", "location": "A steel pedestrian bridge.", "default_background": "City lights beyond the railing."}}],
    "styles": [{"id": "style_anime", "display_name": "日系手绘", "card": {"style": "Japanese anime illustration."}}],
    "extension_cards": [{"kind": "org.example.prop", "id": "prop_letter", "display_name": "信件",
        "data": {"description": "A sealed letter."}, "extensions": {"org.example.test": {"enabled": True}}}],
})
assert library["characters"][0]["card"]["description"] == "A short girl with long blue hair."
assert library["characters"][0]["instance_defaults"]["pose_override"] == "stands naturally"
assert library["environments"][0]["card"]["location"] == "A steel pedestrian bridge."
assert library["styles"][0]["card"]["style"] == "Japanese anime illustration."
assert library["extension_cards"][0]["data"]["description"] == "A sealed letter."
try:
    mod.resource_library.normalize_library({"characters": [{"display_name": "bad", "card": {"name": "bad"},
        "reference_image": {"type": "input", "subfolder": "../outside", "filename": "bad.png"}}]})
    raise AssertionError("unsafe resource image path was accepted")
except ValueError:
    pass

original_library_path = mod.resource_library._library_path
with tempfile.TemporaryDirectory() as resource_temp:
    mod.resource_library._library_path = lambda: str(pathlib.Path(resource_temp) / "library.json")
    saved_library = mod.resource_library.save_library(library, 0)
    character_path = pathlib.Path(resource_temp) / "cards" / "character" / "character_luluka.json"
    extension_path = pathlib.Path(resource_temp) / "cards" / "org.example.prop" / "prop_letter.json"
    assert character_path.exists()
    assert extension_path.exists()
    assert json.loads(character_path.read_text(encoding="utf-8"))["format"] == "minimax-h3-resource-card"
    assert json.loads((pathlib.Path(resource_temp) / "library.json").read_text(encoding="utf-8"))["format"] == "minimax-h3-resource-index"
    reloaded_library = mod.resource_library.load_library()
    assert reloaded_library["revision"] == saved_library["revision"]
    assert reloaded_library["characters"][0]["id"] == "character_luluka"
    assert reloaded_library["extension_cards"][0]["kind"] == "org.example.prop"
mod.resource_library._library_path = original_library_path

image = torch.zeros(1, 64, 64, 3)
lang = s.LanguageData("Chinese", "Mandarin Chinese", "standard Mandarin accent", "natural pronunciation")
card = s.CharacterCardData("Luluka", "She is a girl.", "Preserve her identity.", u._reference(image, "character identity"),
    "stands in the center of the frame", "stands naturally", "has a calm expression", "", "", "global")
actor = s.ActorInstanceData(card, "", "", "", "")
group = s.CharacterGroupData((actor,))
style = s.StyleCardData("", "", "", "", None)
env_card = s.EnvironmentCardData("bridge", "", "", "", "", "", None)
env = s.EnvironmentInstanceData(env_card, "", "", "", "")
clip = s.TimelineClipData("body", 0.0, 5.0, "dances", "", "", lang, "", "on-screen", None, "", None)
track = s.TimelineTrackData("actor", actor, (clip,))
tracks = s.TrackListData((track,))
timeline = s.TimelineData(group, style, env, tracks, 5.0)

safe_card = mod.MiniMaxH3Character.execute("Luluka", "A short girl with long blue hair.", "", "global", image)[0]
assert not any((safe_card.default_position, safe_card.default_pose, safe_card.default_emotion,
    safe_card.default_appearance))
assert safe_card.preservation == "Preserve identity and fixed appearance throughout the video."
assert mod.MiniMaxH3ActorInstance.execute(safe_card, actor_id="actor_7")[0].actor_id == "actor_7"
try:
    mod.MiniMaxH3ActorInstance.execute(safe_card, actor_id="hero")
    raise AssertionError("Invalid actor instance macro was accepted")
except ValueError as error:
    assert "actor_1" in str(error)
safe_environment = mod.MiniMaxH3Environment.execute("bridge", "A steel pedestrian bridge.",
    "Fixed railings line both sides.", image)[0]
assert not safe_environment.default_time_weather and not safe_environment.default_atmosphere
assert safe_environment.location == "A steel pedestrian bridge."

motion_frames = torch.arange(48, dtype=torch.float32).reshape(48, 1, 1, 1).expand(-1, 16, 16, 3)
motion_audio = {"waveform": torch.zeros(1, 2, 64000), "sample_rate": 32000}
source = s.ReferenceVideoData(motion_frames, motion_audio, 2.0)
motion = s.ActorPerformanceReferenceData(source)
aligned_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "follows the dance", motion_reference=motion)[0]
assert aligned_clip.motion_reference is motion
audio_context_action = mod.MiniMaxH3Action.execute("body", 2.0, 4.0, "continues speaking",
    use_previous_context=False, audio_only_context=True)[0]
assert not audio_context_action.use_previous_context and audio_context_action.audio_only_context
context_timeline = s.TimelineData(group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", actor, (
        mod.MiniMaxH3Action.execute("body", 0.0, 2.0, "walks", result="PREVIOUS_VISUAL_STATE_MUST_NOT_LEAK")[0],
        audio_context_action)),)), 4.0)
assert mod.segments._segment_context_mode(context_timeline, 1) == "audio"
audio_context_job = s.GenerationJobData(context_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 2.0, "不输出")
audio_context_plan = mod.segments._segment_frame_plan(2.0, 48, 48)
audio_context_prompt = mod.segments._compile_generation_segment(audio_context_job, 1, audio_context_plan)[0].text
assert "PREVIOUS_VISUAL_STATE_MUST_NOT_LEAK" not in audio_context_prompt
assert mod.timeline._summary_clause("First action sentence. A future event must not enter the summary.") == "First action sentence"
legacy_action = mod.MiniMaxH3Action.execute("body", 0.0, 3.0, "跳舞", quality="旧结束状态",
    result="旧说话方式", use_previous_context="旧动作质量")[0]
assert legacy_action.use_previous_context is True
assert legacy_action.quality == "旧动作质量."
assert legacy_action.result == "旧结束状态."
assert legacy_action.delivery == "旧说话方式"
aligned_track = s.TimelineTrackData("actor", actor, (aligned_clip,))
aligned_timeline = s.TimelineData(group, style, env, s.TrackListData((aligned_track,)), 4.0)
style_reference = s.StyleCardData("", "", "", "", u._reference(torch.ones_like(image), "visual style"))
referenced_timeline = s.TimelineData(group, style_reference, env, s.TrackListData((aligned_track,)), 4.0)
assert mod.segments._reference_subject_count(referenced_timeline) == 2
aligned_video_groups, aligned_audio_groups = mod.segments._reference_media(aligned_timeline)
aligned_instructions, aligned_definitions, aligned_retentions, aligned_summary, aligned_audios = mod.segments._semantic_references(
    aligned_timeline, aligned_video_groups, aligned_audio_groups, 2)
assert len(aligned_video_groups) == 1 and not aligned_audios
aligned_reference = aligned_video_groups[0][1]
assert aligned_reference.frames.shape[0] == 107
assert torch.equal(aligned_reference.frames[0], motion_frames[0])
assert torch.equal(aligned_reference.frames[-1], motion_frames[-1])
assert torch.all(aligned_reference.frames[96:] == motion_frames[-1])
assert aligned_reference.motion_duration == 4.0 and aligned_reference.audio is None
aligned_instruction = next(iter(aligned_instructions.values()))
assert "authoritative body-performance" in aligned_instruction
assert "complete motion order" in aligned_instruction
assert "format padding" not in aligned_instruction
assert "<Subject 2> is the body performance derived from <Video 1> and transferred to <Subject 1> ({actor_1})." in aligned_definitions
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in aligned_retentions
assert aligned_summary == "Body performance is transferred from <Subject 2>."

full_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "follows the complete performance",
    motion_reference=motion)[0]
full_track = s.TimelineTrackData("actor", actor, (full_clip,))
camera_clip = mod.MiniMaxH3Camera.execute(0.0, 4.0, "A low-angle wide shot frames {actor_1}.",
    "The camera orbits clockwise.", "Deep focus keeps the stage sharp.")[0]
camera_track = s.TimelineTrackData("camera", None, (camera_clip,))
referenced_camera_clip = mod.MiniMaxH3Camera.execute(0.0, 4.0, "Conflicting camera text.",
    "Conflicting camera movement.", "Conflicting focus.", camera_reference=s.CameraReferenceData(source))[0]
referenced_camera_track = s.TimelineTrackData("camera", None, (referenced_camera_clip,))
referenced_lighting_clip = mod.MiniMaxH3LightingAction.execute(0.0, 4.0, "Conflicting lighting text.",
    lighting_reference=s.LightingReferenceData(source))[0]
referenced_lighting_track = s.TimelineTrackData("lighting", None, (referenced_lighting_clip,))
referenced_audio_clip = mod.MiniMaxH3AudioAction.execute("music", 0.0, 4.0, "A rhythmic score.",
    audio_reference=s.AudioReferenceData(source))[0]
referenced_audio_track = s.TimelineTrackData("audio", None, (referenced_audio_clip,))
full_timeline = s.TimelineData(group, style, env, s.TrackListData((full_track, referenced_camera_track,
    referenced_lighting_track, referenced_audio_track)), 4.0)
full_job = s.GenerationJobData(full_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出")
full_compiled, full_references, full_standalone_audios = mod.segments._compile_generation_segment(full_job, 0)
full_prompt = full_compiled.text
assert len(full_references) == 1 and not full_standalone_audios
assert full_references[0].role == "actor + camera + lighting"
assert full_references[0].audio is not None
assert "stands in the center of the frame" not in full_prompt
assert "stands naturally" not in full_prompt
assert "has a calm expression" not in full_prompt
assert "<Subject 2> is the body performance derived from <Video 1> and transferred to <Subject 1> (Luluka)." in full_prompt
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in full_prompt
assert "<Subject 3> is the lighting behavior derived from <Video 1>" in full_prompt
assert "<Video 1> is the reference for [Shot 1]'s camera movement and framing progression." in full_prompt
assert "<Video 1> (temporal structure in [Shot 1]): fully_preserved" in full_prompt
assert "Camera movement and temporal structure follow <Video 1>." in full_prompt
assert "<Audio 1> is the music sound reference from the soundtrack associated with <Video 1>." in full_prompt
assert "A low-angle wide shot frames Luluka" not in full_prompt
assert "Conflicting camera" not in full_prompt and "Conflicting lighting" not in full_prompt
assert "Motion references:" not in full_prompt
assert "authoritative camera movement" in full_prompt and "authoritative lighting behavior" in full_prompt

shared_body = mod.MiniMaxH3Action.execute("body", 0.0, 4.0,
    "{actor_1} follows person_1's body movement.", motion_reference=s.ActorPerformanceReferenceData(source))[0]
shared_expression = mod.MiniMaxH3Action.execute("expression", 0.0, 4.0,
    "{actor_1} follows {person_1}'s facial performance.", motion_reference=s.ActorPerformanceReferenceData(source))[0]
shared_camera = mod.MiniMaxH3Camera.execute(0.0, 4.0, "", "", "",
    camera_reference=s.CameraReferenceData(source))[0]
shared_lighting = mod.MiniMaxH3LightingAction.execute(0.0, 4.0, "",
    lighting_reference=s.LightingReferenceData(source))[0]
shared_environment = mod.MiniMaxH3EnvironmentAction.execute(0.0, 4.0, "",
    environment_reference=s.EnvironmentReferenceData(source))[0]
shared_audio = mod.MiniMaxH3AudioAction.execute("music", 0.0, 4.0, "",
    audio_reference=s.AudioReferenceData(source))[0]
shared_timeline = s.TimelineData(group, style, env, s.TrackListData((
    s.TimelineTrackData("actor", actor, (shared_body, shared_expression)),
    s.TimelineTrackData("camera", None, (shared_camera,)),
    s.TimelineTrackData("lighting", None, (shared_lighting,)),
    s.TimelineTrackData("environment", env, (shared_environment,)),
    s.TimelineTrackData("audio", None, (shared_audio,)))), 4.0, "中文")
shared_job = s.GenerationJobData(shared_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出")
shared_compiled, shared_references, shared_audios = mod.segments._compile_generation_segment(shared_job, 0)
shared_prompt = shared_compiled.text
assert len(shared_references) == 1 and not shared_audios
assert shared_references[0].role == "actor + camera + lighting + environment"
assert "面部表演" in shared_prompt and "权威面部表演参考" in shared_prompt
assert "只替换源人物" in shared_prompt
assert "<Subject 4>（出现在[Shot 1]）：fully_preserved" in shared_prompt
assert "{person_1}" not in shared_prompt and "person_1" in shared_prompt
assert "<Audio 1>：reference" in shared_prompt and "fully_copy" not in shared_prompt

reference_only_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "", motion_reference=motion)[0]
reference_only_timeline = mod.MiniMaxH3Timeline.execute(group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", actor, (reference_only_clip,)),)), 4.0)[0]
assert len(reference_only_timeline.tracks.tracks[0].clips) == 1
reference_only_prompt = mod.segments._compile_generation_segment(
    s.GenerationJobData(reference_only_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出"), 0)[0].text
assert "authoritative body-performance" in reference_only_prompt

standalone_audio_clip = mod.MiniMaxH3AudioAction.execute("music", 0.0, 4.0, "Follow the supplied rhythm.",
    audio_reference=s.AudioReferenceData(source))[0]
standalone_audio_timeline = s.TimelineData(group, style, env,
    s.TrackListData((s.TimelineTrackData("audio", None, (standalone_audio_clip,)),)), 4.0)
standalone_compiled, standalone_videos, standalone_audios = mod.segments._compile_generation_segment(
    s.GenerationJobData(standalone_audio_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出"), 0)
assert not standalone_videos and len(standalone_audios) == 1
assert "<Audio 1> is the music sound reference" in standalone_compiled.text
assert "<Video 1>" not in standalone_compiled.text
assert "[reference generation + audio reference]" in standalone_compiled.text

action_camera_timeline = s.TimelineData(group, style, env, s.TrackListData((aligned_track, camera_track)), 4.0)
action_camera_job = s.GenerationJobData(action_camera_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92,
    "不输出")
action_camera_prompt = mod.segments._compile_generation_segment(action_camera_job, 0)[0].text
assert "A low-angle wide shot frames Luluka" in action_camera_prompt
assert "The camera orbits clockwise" in action_camera_prompt

overridden_actor = s.ActorInstanceData(card, "starts beside the window", "", "", "")
overridden_group = s.CharacterGroupData((overridden_actor,))
overridden_track = s.TimelineTrackData("actor", overridden_actor, (full_clip,))
overridden_timeline = s.TimelineData(overridden_group, style, env, s.TrackListData((overridden_track,)), 4.0)
overridden_job = s.GenerationJobData(overridden_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92,
    "不输出")
overridden_prompt = mod.segments._compile_generation_segment(overridden_job, 0)[0].text
assert "starts beside the window" not in overridden_prompt
assert "stands naturally" not in overridden_prompt
assert "has a calm expression" not in overridden_prompt

idle_timeline = s.TimelineData(overridden_group, style, env, s.TrackListData(()), 4.0)
idle_prompt = mod.MiniMaxH3FinalPrompt.execute(idle_timeline, 0.4, "16:9", "Ref", None, None, "")[0].text
assert "starts beside the window" in idle_prompt

stateful_environment = s.EnvironmentInstanceData(
    s.EnvironmentCardData("bridge", "A steel pedestrian bridge.", "Legacy stormy night.",
        "Fixed railings line both sides.", "Legacy tense atmosphere.", "", None),
    "", "A clear morning.", "", "Quiet and open.")
environment_prompt = mod.MiniMaxH3FinalPrompt.execute(
    s.TimelineData(group, style, stateful_environment, s.TrackListData(()), 4.0),
    0.4, "16:9", "Ref", None, None, "")[0].text
assert "A steel pedestrian bridge" in environment_prompt
assert "Fixed railings line both sides" in environment_prompt
assert "A clear morning" in environment_prompt and "Quiet and open" in environment_prompt
assert "Legacy stormy night" not in environment_prompt and "Legacy tense atmosphere" not in environment_prompt

result_components = mod.checkpoints.Types.VideoComponents(torch.zeros(150, 32, 48, 3), Fraction(30),
    {"waveform": torch.zeros(1, 1, 220500), "sample_rate": 44100})
result_video = mod.checkpoints.InputImpl.VideoFromComponents(result_components)
split_motion, split_camera, split_lighting, split_audio, split_environment = mod.MiniMaxH3MotionReference.execute(
    result_video, 1.0, 3.0)
assert split_motion.source.frames.shape[0] == 48
assert split_motion.source is split_camera.source is split_lighting.source is split_audio.source is split_environment.source
assert split_motion.source.audio["waveform"].shape[-1] == 88200
left_motion = mod.MiniMaxH3ReferenceVideoPerson.execute(
    split_motion, "person_1", "the performer on the left wearing a white shirt")[0]
right_motion = mod.MiniMaxH3ReferenceVideoPerson.execute(
    split_motion, "person_2", "the performer on the right wearing a black jacket")[0]
assert left_motion.source is right_motion.source is split_motion.source
assert left_motion.person_id == "person_1" and "on the left" in left_motion.person_description
second_card = s.CharacterCardData("Yona", "A young woman with short black hair.", "Preserve her identity.",
    u._reference(torch.ones_like(image), "character identity"), "", "", "", "", "", "global")
second_actor = s.ActorInstanceData(second_card, "", "", "", "", "actor_2")
multi_group = s.CharacterGroupData((actor, second_actor))
left_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "dances on the left", motion_reference=left_motion)[0]
right_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "dances on the right", motion_reference=right_motion)[0]
multi_timeline = s.TimelineData(multi_group, style, env, s.TrackListData((
    s.TimelineTrackData("actor", actor, (left_clip,)),
    s.TimelineTrackData("actor", second_actor, (right_clip,)))), 4.0)
multi_prompt = mod.segments._compile_generation_segment(
    s.GenerationJobData(multi_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.0, "不输出"), 0)[0].text
assert "source performer person_1 (the performer on the left wearing a white shirt)" in multi_prompt
assert "source performer person_2 (the performer on the right wearing a black jacket)" in multi_prompt
assert "follow only source performer person_1" in multi_prompt
assert "follow only source performer person_2" in multi_prompt
assert mod.reference_assets._normalize_people([
    {"id": "person_1", "description": "left performer"},
    {"id": "person_2", "description": "right performer"},
]) == [{"id": "person_1", "description": "left performer"}, {"id": "person_2", "description": "right performer"}]
result_clip = mod.MiniMaxH3ActionResult.execute(clip, result_video, 2)[0]
result_track = s.TimelineTrackData("actor", actor, (result_clip,))
result_timeline = s.TimelineData(group, style, env, s.TrackListData((result_track,)), 5.0)
assert mod.segments._segment_result(result_timeline, 0.0, 5.0) == (result_video, 2)
prepared_images, prepared_audio = mod.MiniMaxH3SegmentResultPrepare.execute(result_video, 5.0, 64, 64)[:2]
assert prepared_images.shape == (120, 64, 64, 3)
assert prepared_audio["waveform"].shape == (1, 2, 160000)
assert prepared_audio["sample_rate"] == 32000


second_pass_source = torch.zeros(5, 32, 64, 4)
second_pass_resized = mod.MiniMaxH3SecondPassResize.execute(second_pass_source, 0.01, "16:9")[0]
second_pass_width, second_pass_height = u._video_size(0.01, "16:9")
assert second_pass_resized.shape == (5, second_pass_height, second_pass_width, 3)

omit = mod.MiniMaxH3FinalPrompt.execute(timeline, 0.98, "16:9", "Ref", None, None, "",
    empty_sections="不输出")[0].text
na = mod.MiniMaxH3FinalPrompt.execute(timeline, 0.98, "16:9", "Ref", None, None, "",
    empty_sections="输出 N/A")[0].text

sections = ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
    "overall_soundscape:", "non_diegetic_music:")
assert all(section in omit for section in sections)
assert [omit.index(section) for section in sections] == sorted(omit.index(section) for section in sections)
assert "Natural environmental ambience" in omit
assert "overall_soundscape:\nN/A" in na
assert "non_diegetic_music:\nN/A" in na
assert "Strict chronological timeline" not in na
assert "Do not anticipate" not in na
assert "Main visible action: Luluka dances." in na

direct_name_clip = s.TimelineClipData("body", 0.0, 5.0, "Luluka dances", "", "", lang, "", "on-screen", None, "", None)
direct_name_timeline = s.TimelineData(group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", actor, (direct_name_clip,)),)), 5.0)
direct_name_prompt = mod.MiniMaxH3FinalPrompt.execute(
    direct_name_timeline, 0.98, "16:9", "Ref", None, None, "")[0].text
assert "Luluka dances." in direct_name_prompt and "Luluka Luluka dances" not in direct_name_prompt
invalid_clip = s.TimelineClipData("body", 0.0, 5.0, "{actor_2} dances", "", "", lang, "", "on-screen", None, "", None)
invalid_timeline = s.TimelineData(group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", actor, (invalid_clip,)),)), 5.0)
try:
    mod.MiniMaxH3FinalPrompt.execute(invalid_timeline, 0.98, "16:9", "Ref", None, None, "")
    raise AssertionError("Undeclared actor macro was accepted")
except ValueError as error:
    assert "未声明的人物宏" in str(error)

audio_clip = s.TimelineClipData("audio", 0.0, 5.0, "Steady rain ambience", "", "", None, "", "on-screen",
    None, "ambience", None, None, 0)
audio_track = s.TimelineTrackData("audio", None, (audio_clip,))
audio_timeline = s.TimelineData(group, style, env, s.TrackListData((track, audio_track)), 5.0)
audio_prompt = mod.MiniMaxH3FinalPrompt.execute(audio_timeline, 0.98, "16:9", "Ref", None, None, "",
    empty_sections="输出 N/A")[0].text
assert audio_prompt.count("Steady rain ambience") == 1
assert "overall_soundscape:\nSteady rain ambience." in audio_prompt

first_action = s.TimelineClipData("body", 0.0, 2.5, "walks to the window", "", "", lang, "", "on-screen",
    None, "", None, None, 0)
second_action = s.TimelineClipData("body", 2.5, 5.0, "looks outside", "", "", lang, "", "on-screen",
    None, "", None, None, 0)
ordered_track = s.TimelineTrackData("actor", actor, (first_action, second_action))
ordered_timeline = s.TimelineData(group, style, env, s.TrackListData((ordered_track,)), 5.0)
ordered_prompt = mod.MiniMaxH3FinalPrompt.execute(ordered_timeline, 0.98, "16:9", "Ref", None, None, "",
    empty_sections="输出 N/A")[0].text
assert ordered_prompt.index("walks to the window") < ordered_prompt.index("Then, from 2.5 to 5 seconds")
assert "Do not" not in ordered_prompt
assert "Luluka (S1)" not in ordered_prompt

short_plan = mod.segments._segment_frame_plan(0.92, 124, 96)
assert (short_plan.requested_frames, short_plan.current_frames, short_plan.locked_frames,
    short_plan.generation_frames, short_plan.trailing_frames) == (96, 107, 22, 141, 12)
long_plan = mod.segments._segment_frame_plan(2.0, 124, 96)
assert (long_plan.requested_frames, long_plan.current_frames, long_plan.locked_frames,
    long_plan.generation_frames, long_plan.trailing_frames) == (96, 107, 39, 158, 12)
first_plan = mod.segments._segment_frame_plan(2.0, 0, 96)
assert (first_plan.requested_frames, first_plan.current_frames, first_plan.locked_frames,
    first_plan.generation_frames) == (96, 107, 0, 107)
assert mod.segments._segment_frame_plan(2.0, 40, 96) == long_plan
pv_ranges = ((0.0, 4.4), (4.4, 8.8), (8.8, 13.2), (13.2, 17.6), (17.6, 22.0))
assert [mod.segments._segment_visible_frames(*item) for item in pv_ranges] == [106, 105, 106, 105, 106]
assert sum(mod.segments._segment_visible_frames(*item) for item in pv_ranges) == 528
assert mod.segments._empty_sections_mode("不输出") == "自动补全"

previous_source = s.ReferenceVideoData(torch.ones(107, 16, 16, 3),
    {"waveform": torch.ones(1, 2, 142667), "sample_rate": 32000}, 4.0)
current_source = s.ReferenceVideoData(torch.full((107, 16, 16, 3), 2.0),
    {"waveform": torch.full((1, 2, 142667), 2.0), "sample_rate": 32000}, 4.0)
previous_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "walks forward",
    motion_reference=s.ActorPerformanceReferenceData(previous_source))[0]
current_clip = mod.MiniMaxH3Action.execute("body", 4.0, 8.0, "performs the dance",
    motion_reference=s.ActorPerformanceReferenceData(current_source))[0]
segmented_track = s.TimelineTrackData("actor", actor, (previous_clip, current_clip))
previous_audio_clip = mod.MiniMaxH3AudioAction.execute("music", 0.0, 4.0, "Previous rhythmic score.",
    audio_reference=s.AudioReferenceData(previous_source))[0]
current_audio_clip = mod.MiniMaxH3AudioAction.execute("music", 4.0, 8.0, "Current rhythmic score.",
    audio_reference=s.AudioReferenceData(current_source))[0]
segmented_audio_track = s.TimelineTrackData("audio", None, (previous_audio_clip, current_audio_clip))
segmented_timeline = s.TimelineData(group, style, env, s.TrackListData((segmented_track, segmented_audio_track)), 8.0)
job = s.GenerationJobData(segmented_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 2.0, "不输出")
compiled_segment, segment_references, segment_audios = mod.segments._compile_generation_segment(job, 1, long_plan)
assert not segment_audios
segment_reference = segment_references[0]
assert compiled_segment.video_settings.length == 158
assert segment_reference.frames.shape[0] == 158
assert torch.all(segment_reference.frames[:39] == 1)
assert torch.all(segment_reference.frames[39:] == 2)
assert segment_reference.context_duration == 39 / 24
assert segment_reference.locked_duration == 39 / 24
assert segment_reference.audio["waveform"].shape[-1] == 210667
assert torch.all(segment_reference.audio["waveform"][..., :52000] == 1)
assert torch.all(segment_reference.audio["waveform"][..., 52000:194667] == 2)
assert torch.all(segment_reference.audio["waveform"][..., 194667:] == 0)
try:
    mod.segments._validate_motion_alignment((s.MotionReferenceData(
        segment_reference.frames[:-1], None, "actor", 4.0, 157 / 24),), 158)
    raise AssertionError("Motion reference length mismatch was accepted")
except ValueError as error:
    assert "157 帧" in str(error) and "158 帧" in str(error)
assert "The target video is a 6.58-second continuous single shot." in compiled_segment.text
assert "[video continuation + reference generation + audio reference]" in compiled_segment.text
assert "<Subject 2> is the body performance derived from <Video 1>" in compiled_segment.text
assert "<Video 1> is the reference for [Shot 1]'s motion-transition context and current shot-aligned temporal order" in compiled_segment.text
assert "The opening 1.62 seconds are hard-locked to the preceding generated segment" in compiled_segment.text
assert "At 1.62 seconds, the current action continues directly from the locked final frame" in compiled_segment.text
assert "Use <Subject 2> as <Subject 1> (Luluka)'s authoritative body-performance" in compiled_segment.text
assert "Transfer only body performance; do not copy source performer" in compiled_segment.text
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in compiled_segment.text
assert "Motion references:" not in compiled_segment.text
details = compiled_segment.text.split("detailed_description:\n", 1)[1]
assert details.index("The opening 1.62 seconds") < details.index("From 1.62 to 6.08 seconds, Luluka performs the dance")
assert "From 6.08 to 6.58 seconds, the final visible state" in details

speech_boundary = mod.MiniMaxH3Action.execute("speech", 1.0, 3.0, "你好", language=lang)[0]
body_driven_track = s.TimelineTrackData("actor", actor, (previous_clip, current_clip, speech_boundary))
body_driven_timeline = s.TimelineData(group, style, env, s.TrackListData((body_driven_track,)), 8.0)
assert mod.segments._segment_ranges(body_driven_timeline) == ((0.0, 4.0), (4.0, 8.0))

current_without_context = mod.MiniMaxH3Action.execute("body", 4.0, 8.0, "performs the dance",
    motion_reference=s.ActorPerformanceReferenceData(current_source), use_previous_context=False)[0]
no_context_track = s.TimelineTrackData("actor", actor, (previous_clip, current_without_context))
no_context_timeline = s.TimelineData(group, style, env, s.TrackListData((no_context_track,)), 8.0)
no_context_job = s.GenerationJobData(no_context_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 2.0, "不输出")
no_context_ranges = mod.segments._segment_ranges(no_context_timeline)
assert mod.segments._segment_continuity_seconds(no_context_job, 1, no_context_ranges) == 0.0
no_context_plan = mod.segments._segment_frame_plan(0.0, 96, 96)
no_context_compiled, no_context_references, _ = mod.segments._compile_generation_segment(
    no_context_job, 1, no_context_plan)
assert no_context_plan.locked_frames == 0
assert no_context_references[0].context_duration == 0.0
assert "hard-locked to the preceding generated segment" not in no_context_compiled.text
tail_source = torch.arange(12).reshape(12, 1, 1, 1)
assert torch.equal(mod.segments._tail_align_frames(tail_source, 5).flatten(), torch.arange(7, 12))

previous_without_reference = s.TimelineClipData("body", 0.0, 4.0, "walks forward", "", "", lang, "",
    "on-screen", None, "", None)
fallback_track = s.TimelineTrackData("actor", actor, (previous_without_reference, current_clip))
fallback_timeline = s.TimelineData(group, style, env, s.TrackListData((fallback_track, segmented_audio_track)), 8.0)
fallback_job = s.GenerationJobData(fallback_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 2.0, "不输出")
generated_tail = torch.full((96, 16, 16, 3), 3.0)
generated_audio = {"waveform": torch.full((1, 2, 128000), 3.0), "sample_rate": 32000}
fallback_compiled, fallback_references, fallback_audios = mod.segments._compile_generation_segment(fallback_job, 1, long_plan,
    generated_tail, generated_audio)
assert not fallback_audios
fallback_reference = fallback_references[0]
assert torch.all(fallback_reference.frames[:39] == 3)
assert torch.all(fallback_reference.frames[39:] == 2)
assert torch.all(fallback_reference.audio["waveform"][..., :52000] == 3)
assert "[video continuation + reference generation + audio reference]" in fallback_compiled.text
assert "The opening 1.62 seconds are hard-locked to the preceding generated segment" in fallback_compiled.text

current_without_reference = s.TimelineClipData("body", 4.0, 8.0, "walks toward the door", "", "", lang, "",
    "on-screen", None, "", None)
latent_only_track = s.TimelineTrackData("actor", actor, (previous_clip, current_without_reference))
latent_only_job = s.GenerationJobData(
    s.TimelineData(group, style, env, s.TrackListData((latent_only_track,)), 8.0),
    0.4, "16:9", 0, "simple", 4, 1.0, "match", 2.0, "不输出")
latent_only_compiled, latent_only_references, latent_only_audios = mod.segments._compile_generation_segment(latent_only_job, 1,
    long_plan)
assert not latent_only_references
assert not latent_only_audios
assert "[video continuation + reference generation]" in latent_only_compiled.text
assert "The opening 1.62 seconds are hard-locked to the preceding generated segment" in latent_only_compiled.text

class VideoVAE:
    def encode(self, frames):
        return torch.ones(1, 24, 7, 4, 4)

class AudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        return torch.ones(1, 32, 2, 37)

latent = {"samples": mod.segments.comfy.nested_tensor.NestedTensor((
    torch.zeros(1, 24, 12, 4, 4), torch.zeros(1, 32, 2, 100)))}
previous_images = torch.zeros(30, 64, 64, 3)
previous_audio = {"waveform": torch.zeros(1, 2, 40000), "sample_rate": 32000}
locked = mod.segments._lock_context_prefix(latent, previous_images, previous_audio, None,
    VideoVAE(), AudioVAE(), 22, 64, 64)
video, audio = locked["samples"].unbind()
video_mask, audio_mask = locked["noise_mask"].unbind()
assert torch.all(video[:, :, :7] == 1)
assert torch.all(video_mask[:, :, :7] == 0) and torch.all(video_mask[:, :, 7:] == 1)
assert torch.all(audio[..., :37] == 1)
assert torch.all(audio_mask[..., :37] == 0) and torch.all(audio_mask[..., 37:] == 1)

audio_locked = mod.segments._lock_context_prefix(latent, previous_images, previous_audio, None,
    VideoVAE(), AudioVAE(), 22, 64, 64, lock_video=False)
audio_only_video, audio_only_audio = audio_locked["samples"].unbind()
audio_only_video_mask, audio_only_audio_mask = audio_locked["noise_mask"].unbind()
assert torch.all(audio_only_video == 0) and torch.all(audio_only_video_mask == 1)
assert torch.all(audio_only_audio[..., :37] == 1)
assert torch.all(audio_only_audio_mask[..., :37] == 0) and torch.all(audio_only_audio_mask[..., 37:] == 1)

class UnusedVAE:
    def encode(self, value):
        raise AssertionError("Direct Latent continuity unexpectedly re-encoded decoded media")

previous_video_latent = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20, 1, 1).repeat(1, 24, 1, 4, 4)
previous_audio_latent = torch.arange(60, dtype=torch.float32).reshape(1, 1, 1, 60).repeat(1, 32, 2, 1)
direct_previous = {"samples": mod.segments.comfy.nested_tensor.NestedTensor((
    previous_video_latent, previous_audio_latent))}
visible_locked = mod.segments._lock_context_prefix(latent, previous_images, previous_audio, direct_previous,
    VideoVAE(), AudioVAE(), 22, 64, 64)
visible_video, visible_audio = visible_locked["samples"].unbind()
assert torch.all(visible_video[:, :, :7] == 1)
assert torch.all(visible_audio[..., :37] == 1)

direct_locked = mod.segments._lock_context_prefix(latent, None, None, direct_previous,
    UnusedVAE(), UnusedVAE(), 22, 64, 64)
direct_video, direct_audio = direct_locked["samples"].unbind()
assert torch.equal(direct_video[:, :, :7], previous_video_latent[:, :, -7:])
assert torch.equal(direct_audio[..., :37], previous_audio_latent[..., -37:])

try:
    mod.MiniMaxH3SegmentTrim.execute(torch.zeros(140, 16, 16, 3),
        {"waveform": torch.zeros(1, 2, 1000), "sample_rate": 32000}, 0, 4.0, 141)
    raise AssertionError("Decoded video length mismatch was accepted")
except ValueError as error:
    assert "140 帧" in str(error) and "141 帧" in str(error)

trim_source = torch.arange(141, dtype=torch.float32).reshape(141, 1, 1, 1).repeat(1, 16, 16, 3)
trim_audio = {"waveform": torch.arange(188000, dtype=torch.float32).reshape(1, 1, -1).repeat(1, 2, 1),
    "sample_rate": 32000}
trimmed_images, trimmed_audio = mod.MiniMaxH3SegmentTrim.execute(
    trim_source, trim_audio, 39, 102 / 24, 141)[:2]
assert trimmed_images.shape[0] == 102
assert trimmed_images[0, 0, 0, 0] == 39 and trimmed_images[-1, 0, 0, 0] == 140
assert trimmed_audio["waveform"].shape[-1] == 136000
assert trimmed_audio["waveform"][0, 0, 0] == 52000

short_trim_audio = {"waveform": torch.ones(1, 2, 187999), "sample_rate": 32000}
short_trimmed = mod.MiniMaxH3SegmentTrim.execute(
    trim_source, short_trim_audio, 39, 102 / 24, 141)[1]
assert short_trimmed["waveform"].shape[-1] == 136000
assert short_trimmed["waveform"][..., -2].eq(1).all() and short_trimmed["waveform"][..., -1].eq(0).all()

long_trim_audio = {"waveform": torch.ones(1, 2, 188100), "sample_rate": 32000}
long_trimmed = mod.MiniMaxH3SegmentTrim.execute(
    trim_source, long_trim_audio, 39, 102 / 24, 141)[1]
assert long_trimmed["waveform"].shape[-1] == 136000

joined_images, joined_audio = mod.MiniMaxH3SegmentJoin.execute(
    trimmed_images, trimmed_audio, trimmed_images, trimmed_audio)[:2]
assert joined_images.shape[0] == 204 and joined_audio["waveform"].shape[-1] == 272000
rounded_images, rounded_audio = mod.MiniMaxH3SegmentJoin.execute(
    trimmed_images, trimmed_audio, trimmed_images,
    {**trimmed_audio, "waveform": trimmed_audio["waveform"][..., :-1]})[:2]
assert rounded_images.shape[0] == 204 and rounded_audio["waveform"].shape[-1] == 272000

three_second_images = torch.zeros(73, 1, 1, 3)
five_second_images = torch.zeros(136, 1, 1, 3)
three_second_audio = {"waveform": torch.zeros(1, 2, 97333), "sample_rate": 32000}
five_second_audio = {"waveform": torch.zeros(1, 2, 181333), "sample_rate": 32000}
rounded_images, rounded_audio = mod.MiniMaxH3SegmentJoin.execute(
    three_second_images, three_second_audio, five_second_images, five_second_audio)[:2]
assert rounded_images.shape[0] == 209 and rounded_audio["waveform"].shape[-1] == 278667
try:
    mod.MiniMaxH3SegmentJoin.execute(trimmed_images, trimmed_audio, trimmed_images,
        {**trimmed_audio, "waveform": trimmed_audio["waveform"][..., :-9]})
    raise AssertionError("Mismatched audio and video lengths were accepted")
except ValueError as error:
    assert "当前段音频长度与画面帧数不一致" in str(error)

old_output = mod.checkpoints.folder_paths.get_output_directory()
with tempfile.TemporaryDirectory() as temp_output:
    mod.checkpoints.folder_paths.set_output_directory(temp_output)
    try:
        cache_name = "segment_001_0123456789abcdef01234567.mp4"
        components = mod.checkpoints.Types.VideoComponents(torch.zeros(5, 32, 32, 3),
            Fraction(24), {"waveform": torch.zeros(1, 2, 7000), "sample_rate": 32000})
        video = mod.checkpoints.InputImpl.VideoFromComponents(components)
        cached_latent = {"samples": mod.segments.comfy.nested_tensor.NestedTensor((
            torch.ones(1, 24, 12, 4, 4), torch.ones(1, 32, 2, 100)))}
        saved_output = mod.MiniMaxH3SegmentCheckpoint.execute(video, cache_name, f'["{cache_name}"]', cached_latent)
        saved = saved_output[0]
        assert mod.checkpoints._cache_path(cache_name).is_file()
        assert mod.checkpoints._latent_path(cache_name).is_file()
        loaded_output = mod.MiniMaxH3SegmentCheckpointLoad.execute(cache_name, f'["{cache_name}"]')
        loaded = loaded_output[0]
        loaded_video_latent, loaded_audio_latent = loaded_output[1]["samples"].unbind()
        assert saved.get_dimensions() == loaded.get_dimensions() == (32, 32)
        assert torch.equal(loaded_video_latent, torch.ones(1, 24, 12, 4, 4))
        assert torch.equal(loaded_audio_latent, torch.ones(1, 32, 2, 100))
    finally:
        mod.checkpoints.folder_paths.set_output_directory(old_output)

empty_card = s.CharacterCardData("", "", "", None, "", "", "", "", "", "global")
empty_actor = s.ActorInstanceData(empty_card, "", "", "", "")
empty_group = s.CharacterGroupData((empty_actor,))
empty_style = s.StyleCardData("", "", "", "", None)
empty_env = s.EnvironmentInstanceData(s.EnvironmentCardData("", "", "", "", "", "", None), "", "", "", "")
empty_timeline = s.TimelineData(empty_group, empty_style, empty_env, s.TrackListData(()), 5.0)
t2va = mod.MiniMaxH3FinalPrompt.execute(empty_timeline, 0.98, "16:9", "FL", None, None, "",
    empty_sections="不输出")[0]
i2va = mod.MiniMaxH3FinalPrompt.execute(empty_timeline, 0.98, "16:9", "FL", image, None, "",
    empty_sections="不输出")[0]
l2va = mod.MiniMaxH3FinalPrompt.execute(empty_timeline, 0.98, "16:9", "FL", None, image, "",
    empty_sections="不输出")[0]
fl2va = mod.MiniMaxH3FinalPrompt.execute(empty_timeline, 0.98, "16:9", "FL", image, image, "",
    empty_sections="不输出")[0]
assert t2va.video_settings.mode == "T2VA"
assert t2va.text.startswith("integrated_multimodal_description:")
assert i2va.video_settings.mode == "I2VA"
assert i2va.text.startswith("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n")
assert l2va.video_settings.mode == "L2VA"
assert "<Picture 1> (from [Shot 1]) aligns with the 5.17-second mark" in l2va.text
assert fl2va.video_settings.mode == "FL2VA"
assert "Picture 1 (from Shot 1) aligns with the 0.00-second mark" in fl2va.text
assert "Picture 2 (from Shot 1) aligns with the 5.17-second mark" in fl2va.text

second_card = s.CharacterCardData("Yona", "She is another girl.", "Preserve her identity.",
    u._reference(image, "character identity"), "", "", "", "", "", "global")
second_actor = s.ActorInstanceData(second_card, "", "", "", "", "actor_2")
shared_group = s.CharacterGroupData((actor, second_actor))
shared_timeline = s.TimelineData(shared_group, style, env, s.TrackListData(()), 5.0)
shared = mod.MiniMaxH3FinalPrompt.execute(shared_timeline, 0.98, "16:9", "Ref", None, None, "")[0]
assert len(shared.references) == 1
assert "<Subject 1> is Luluka, whose identity and appearance come from <Picture 1>." in shared.text
assert "<Subject 2> is Yona, whose identity and appearance come from <Picture 1>." in shared.text

speech = s.TimelineClipData("speech", 0.0, 2.0, "来抱一个。", "", "", lang, "softly",
    "off-screen voiceover", None, "", None)
silent_action = s.TimelineClipData("body", 0.0, 2.0, "turns toward {actor_2}", "", "", lang, "",
    "on-screen", second_actor, "", None)
speaker_tracks = s.TrackListData((s.TimelineTrackData("actor", actor, (silent_action,)),
    s.TimelineTrackData("actor", second_actor, (speech,))))
speaker_timeline = s.TimelineData(shared_group, style, env, speaker_tracks, 5.0)
speaker_prompt = mod.MiniMaxH3FinalPrompt.execute(speaker_timeline, 0.98, "16:9", "Ref", None, None, "")[0].text
assert "<Subject 2> (S1) says in an off-screen voiceover" in speaker_prompt
assert "while Yona's lips remain completely closed" in speaker_prompt
assert "Luluka (S" not in speaker_prompt
assert "retention_analysis:\n<Subject 1> (appears in [Shot 1]):" in speaker_prompt

silent_source = s.ReferenceVideoData(motion_frames, None, 2.0)
silent_motion = s.ActorPerformanceReferenceData(silent_source)
silent_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "follows the dance",
    motion_reference=silent_motion)[0]
silent_motion_timeline = s.TimelineData(group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", actor, (silent_clip,)),)), 4.0)
silent_motion_prompt = mod.segments._compile_generation_segment(
    s.GenerationJobData(silent_motion_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出"), 0)[0].text
assert "<Audio 1>" not in silent_motion_prompt
assert "The target video is a 4.46-second continuous single shot." in silent_motion_prompt
assert "Luluka follows the dance" in silent_motion_prompt
assert "pre-roll" not in silent_motion_prompt
assert "final visible state" not in silent_motion_prompt

class FakeModel:
    model = object()
    patches = {}

    def model_size(self):
        return 1

expanded = mod.MiniMaxH3MultiSegmentGenerate.execute(FakeModel(), object(), object(), object(), object(), job,
    "重新生成全部片段", 0)
assert expanded.expand is not None
assert len(expanded.result) == 2
entry = mod.MiniMaxH3SecondPassEntryPack.execute(object(), 0, 0.0, 5.0, 0, 5, 5, 5 / 24,
    "segment_001_" + "0" * 24 + ".mp4",
    conditioning=object(), latent={"samples": object()})[0]
batch = mod.MiniMaxH3SecondPassBatchAppend.execute(entry)[0]
assert batch.entries == (entry,)
parsed_batch = mod.MiniMaxH3SecondPassBatchParser.execute(batch)
assert parsed_batch[0] == [entry.latent]
assert parsed_batch[1] == [entry.conditioning]
assert parsed_batch[2] == [entry.video]
assert parsed_batch[3] == [1]
assert parsed_batch[4:] == ([0], [5], [5 / 24], [entry.cache_file], [0.0], [5.0], [5])
second_pass = mod.MiniMaxH3MultiSegmentSecondPass.execute(FakeModel(), object(), object(), object(), batch,
    cache_mode="重新生成全部片段")
assert second_pass.expand is not None
latent_second_pass = mod.MiniMaxH3MultiSegmentLatentSecondPass.execute(
    FakeModel(), object(), object(), object(), batch,
    "minimax_h3_latent_upscaler_3d_fp16.safetensors", sigmas=torch.tensor([0.9, 0.0]),
    cache_mode="重新生成全部片段")
assert latent_second_pass.expand is not None
latent_upscale_inputs = next(node["inputs"] for node in latent_second_pass.expand.values()
    if node["class_type"] == "MinimaxH3LatentUpscaler3D")
assert latent_upscale_inputs["mode"] == "megapixels"
assert latent_upscale_inputs["mode.megapixels"] == 1.0

camera_details = action_camera_prompt.split("detailed_description:\n", 1)[1]
assert camera_details.index("A low-angle wide shot frames Luluka") < camera_details.index("Luluka follows the dance")
styled = mod.MiniMaxH3FinalPrompt.execute(
    s.TimelineData(group, s.StyleCardData("2D animation", "", "", "", None), env, tracks, 5.0),
    0.98, "16:9", "Ref", None, None, "")[0].text
assert "detailed_description:\n2D animation.\n[Shot 1]" in styled

chinese_card = s.CharacterCardData("露露卡", "蓝色长发、蓝色眼睛的少女。", "", u._reference(image, "character identity"),
    "画面中央", "自然站立", "神情平静", "", "", "global")
chinese_actor = s.ActorInstanceData(chinese_card, "", "", "", "")
chinese_group = s.CharacterGroupData((chinese_actor,))
chinese_clip = s.TimelineClipData("body", 1.0, 3.0, "向前走两步，然后停下", "动作自然连贯", "保持站立",
    lang, "", "on-screen", None, "", None)
chinese_timeline = s.TimelineData(chinese_group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", chinese_actor, (chinese_clip,)),)), 4.0, "中文")
chinese_prompt = mod.MiniMaxH3FinalPrompt.execute(chinese_timeline, 0.4, "16:9", "Ref", None, None, "")[0].text
assert "<Subject 1>是露露卡，其身份与外观来自<Picture 1>。" in chinese_prompt
assert "目标视频是一个时长" in chinese_prompt
assert "从1秒到3秒，露露卡向前走两步，然后停下" in chinese_prompt
assert "随后，露露卡 保持站立" in chinese_prompt
assert "The target video" not in chinese_prompt
assert "Preserve identity" not in chinese_prompt
chinese_i2va = mod.MiniMaxH3FinalPrompt.execute(
    s.TimelineData(empty_group, empty_style, empty_env, s.TrackListData(()), 5.0, "中文"),
    0.98, "16:9", "FL", image, None, "", empty_sections="输出 N/A")[0].text
assert chinese_i2va.startswith("对于目标视频，在0.00秒处完整参考来自[Shot 1]的<Picture 1>。")
assert "镜头从<Picture 1>开始" in chinese_i2va
assert "The shot begins" not in chinese_i2va
assert mod.MiniMaxH3Timeline.execute(chinese_group, style, env, s.TrackListData(()), 4.0)[0].prompt_language == "英文"
assert mod.MiniMaxH3Timeline.execute(chinese_group, style, env, s.TrackListData(()), 4.0, "中文")[0].prompt_language == "中文"

chinese_motion_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "按照参考动作跳舞",
    motion_reference=silent_motion)[0]
chinese_motion_timeline = s.TimelineData(chinese_group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", chinese_actor, (chinese_motion_clip,)),)), 4.0, "中文")
chinese_motion_prompt = mod.segments._compile_generation_segment(
    s.GenerationJobData(chinese_motion_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出"), 0)[0].text
assert "是从<Video 1>提取并迁移给<Subject 1> (露露卡)的肢体表演" in chinese_motion_prompt
assert "肢体表演迁移自<Subject 2>" in chinese_motion_prompt
assert "权威肢体表演参考" in chinese_motion_prompt
assert "body performance" not in chinese_motion_prompt

print("PASS")
