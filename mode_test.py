import importlib.util
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
assert "authoritative body performance" in aligned_instruction
assert "complete motion order" in aligned_instruction
assert "format padding" not in aligned_instruction
assert "<Subject 2> is the body performance derived from <Video 1> and transferred to <Subject 1> (Luluka)." in aligned_definitions
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in aligned_retentions
assert aligned_summary == "Body performance is transferred from <Subject 2>."

full_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "follows the complete performance",
    motion_reference=motion)[0]
full_track = s.TimelineTrackData("actor", actor, (full_clip,))
camera_clip = mod.MiniMaxH3Camera.execute(0.0, 4.0, "A low-angle wide shot frames Luluka.",
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

reference_only_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "", motion_reference=motion)[0]
reference_only_timeline = mod.MiniMaxH3Timeline.execute(group, style, env,
    s.TrackListData((s.TimelineTrackData("actor", actor, (reference_only_clip,)),)), 4.0)[0]
assert len(reference_only_timeline.tracks.tracks[0].clips) == 1
reference_only_prompt = mod.segments._compile_generation_segment(
    s.GenerationJobData(reference_only_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出"), 0)[0].text
assert "authoritative body performance" in reference_only_prompt

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
split_motion, split_camera, split_lighting, split_audio = mod.MiniMaxH3MotionReference.execute(
    result_video, 1.0, 3.0)
assert split_motion.source.frames.shape[0] == 48
assert split_motion.source is split_camera.source is split_lighting.source is split_audio.source
assert split_motion.source.audio["waveform"].shape[-1] == 88200
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
    short_plan.generation_frames) == (96, 102, 22, 124)
long_plan = mod.segments._segment_frame_plan(2.0, 124, 96)
assert (long_plan.requested_frames, long_plan.current_frames, long_plan.locked_frames,
    long_plan.generation_frames) == (96, 102, 39, 141)
first_plan = mod.segments._segment_frame_plan(2.0, 0, 96)
assert (first_plan.requested_frames, first_plan.current_frames, first_plan.locked_frames,
    first_plan.generation_frames) == (96, 107, 0, 107)
assert mod.segments._segment_frame_plan(2.0, 40, 96) == long_plan
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
assert compiled_segment.video_settings.length == 141
assert segment_reference.frames.shape[0] == 141
assert torch.all(segment_reference.frames[:39] == 1)
assert torch.all(segment_reference.frames[39:] == 2)
assert segment_reference.context_duration == 39 / 24
assert segment_reference.locked_duration == 39 / 24
assert segment_reference.audio["waveform"].shape[-1] == 188000
assert torch.all(segment_reference.audio["waveform"][..., :52000] == 1)
assert torch.all(segment_reference.audio["waveform"][..., 52000:] == 2)
try:
    mod.segments._validate_motion_alignment((s.MotionReferenceData(
        segment_reference.frames[:-1], None, "actor", 4.0, 140 / 24),), 141)
    raise AssertionError("Motion reference length mismatch was accepted")
except ValueError as error:
    assert "140 帧" in str(error) and "141 帧" in str(error)
assert "The target video is a 5.88-second continuous single shot." in compiled_segment.text
assert "[video continuation + reference generation + audio reference]" in compiled_segment.text
assert "<Subject 2> is the body performance derived from <Video 1>" in compiled_segment.text
assert "<Video 1> is the reference for [Shot 1]'s motion-transition context and current shot-aligned temporal order" in compiled_segment.text
assert "The opening 1.62 seconds are hard-locked to the preceding generated segment" in compiled_segment.text
assert "At 1.62 seconds, the current action continues directly from the locked final frame" in compiled_segment.text
assert "Use <Subject 2> as <Subject 1> (Luluka)'s authoritative body performance" in compiled_segment.text
assert "Transfer only body performance; do not copy the source performer" in compiled_segment.text
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in compiled_segment.text
assert "Motion references:" not in compiled_segment.text
details = compiled_segment.text.split("detailed_description:\n", 1)[1]
assert details.index("The opening 1.62 seconds") < details.index("From 1.62 to 5.88 seconds, Luluka performs the dance")

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

class UnusedVAE:
    def encode(self, value):
        raise AssertionError("Direct Latent continuity unexpectedly re-encoded decoded media")

previous_video_latent = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20, 1, 1).repeat(1, 24, 1, 4, 4)
previous_audio_latent = torch.arange(60, dtype=torch.float32).reshape(1, 1, 1, 60).repeat(1, 32, 2, 1)
direct_previous = {"samples": mod.segments.comfy.nested_tensor.NestedTensor((
    previous_video_latent, previous_audio_latent))}
direct_locked = mod.segments._lock_context_prefix(latent, previous_images, previous_audio, direct_previous,
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
second_actor = s.ActorInstanceData(second_card, "", "", "", "")
shared_group = s.CharacterGroupData((actor, second_actor))
shared_timeline = s.TimelineData(shared_group, style, env, s.TrackListData(()), 5.0)
shared = mod.MiniMaxH3FinalPrompt.execute(shared_timeline, 0.98, "16:9", "Ref", None, None, "")[0]
assert len(shared.references) == 1
assert "<Subject 1> is Luluka, whose identity and appearance come from <Picture 1>." in shared.text
assert "<Subject 2> is Yona, whose identity and appearance come from <Picture 1>." in shared.text

speech = s.TimelineClipData("speech", 0.0, 2.0, "来抱一个。", "", "", lang, "softly",
    "off-screen voiceover", None, "", None)
silent_action = s.TimelineClipData("body", 0.0, 2.0, "turns toward Yona", "", "", lang, "",
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

camera_details = action_camera_prompt.split("detailed_description:\n", 1)[1]
assert camera_details.index("A low-angle wide shot frames Luluka") < camera_details.index("Luluka follows the dance")
styled = mod.MiniMaxH3FinalPrompt.execute(
    s.TimelineData(group, s.StyleCardData("2D animation", "", "", "", None), env, tracks, 5.0),
    0.98, "16:9", "Ref", None, None, "")[0].text
assert "detailed_description:\n2D animation.\n[Shot 1]" in styled

print("PASS")
