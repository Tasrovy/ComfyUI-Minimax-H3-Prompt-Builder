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

motion_frames = torch.arange(48, dtype=torch.float32).reshape(48, 1, 1, 1).expand(-1, 16, 16, 3)
motion_audio = {"waveform": torch.zeros(1, 2, 64000), "sample_rate": 32000}
motion = s.MotionReferenceData(motion_frames, motion_audio, "仅动作", 2.0, 2.0)
aligned_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "follows the dance", motion_reference=motion)[0]
assert aligned_clip.motion_reference.frames.shape[0] == 107
assert torch.equal(aligned_clip.motion_reference.frames[0], motion_frames[0])
assert torch.equal(aligned_clip.motion_reference.frames[-1], motion_frames[-1])
assert torch.all(aligned_clip.motion_reference.frames[96:] == motion_frames[-1])
assert aligned_clip.motion_reference.motion_duration == 4.0
assert aligned_clip.motion_reference.audio["waveform"].shape[-1] == round((107 / 24) * 32000)
aligned_track = s.TimelineTrackData("actor", actor, (aligned_clip,))
aligned_timeline = s.TimelineData(group, style, env, s.TrackListData((aligned_track,)), 4.0)
style_reference = s.StyleCardData("", "", "", "", u._reference(torch.ones_like(image), "visual style"))
referenced_timeline = s.TimelineData(group, style_reference, env, s.TrackListData((aligned_track,)), 4.0)
assert mod.segments._reference_subject_count(referenced_timeline) == 2
aligned_references, aligned_instructions, aligned_definitions, aligned_retentions, aligned_summary = mod.segments._motion_references(
    aligned_timeline, 1, 2)
assert len(aligned_references) == 1
assert "Transfer <Subject 2> to Luluka (S1)" in aligned_instructions
assert "complete order, timing, and final pose" in aligned_instructions
assert "format padding" not in aligned_instructions
assert "<Subject 2> is Luluka (S1)'s body motion derived from <Video 1>." in aligned_definitions
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in aligned_retentions
assert aligned_summary == "The action is guided by <Subject 2>."

full_motion = s.MotionReferenceData(motion_frames, motion_audio, "完整表演", 2.0, 2.0)
full_clip = mod.MiniMaxH3Action.execute("body", 0.0, 4.0, "follows the complete performance",
    motion_reference=full_motion)[0]
full_track = s.TimelineTrackData("actor", actor, (full_clip,))
camera_clip = mod.MiniMaxH3Camera.execute(0.0, 4.0, "A low-angle wide shot frames Luluka.",
    "The camera orbits clockwise.", "Deep focus keeps the stage sharp.")[0]
camera_track = s.TimelineTrackData("camera", None, (camera_clip,))
full_timeline = s.TimelineData(group, style, env, s.TrackListData((full_track, camera_track)), 4.0)
full_job = s.GenerationJobData(full_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 0.92, "不输出")
full_prompt = mod.segments._compile_generation_segment(full_job, 0)[0].text
assert "stands in the center of the frame" not in full_prompt
assert "stands naturally" not in full_prompt
assert "has a calm expression" not in full_prompt
assert "<Subject 2> is Luluka (S1)'s complete performance derived from <Video 1>." in full_prompt
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in full_prompt
assert "<Video 1> provides the camera behavior and temporal structure for [Shot 1]." in full_prompt
assert "<Video 1> (camera and temporal structure in [Shot 1]): fully_preserved" in full_prompt
assert "The camera and temporal structure are guided by <Video 1>." in full_prompt
assert "A low-angle wide shot frames Luluka" not in full_prompt
assert "The camera orbits clockwise" not in full_prompt

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
assert "starts beside the window" in overridden_prompt
assert "stands naturally" not in overridden_prompt
assert "has a calm expression" not in overridden_prompt

result_components = mod.checkpoints.Types.VideoComponents(torch.zeros(150, 32, 48, 3), Fraction(30),
    {"waveform": torch.zeros(1, 1, 220500), "sample_rate": 44100})
result_video = mod.checkpoints.InputImpl.VideoFromComponents(result_components)
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

assert "overall_soundscape" not in omit
assert "non_diegetic_music" not in omit
assert "overall_soundscape" in na
assert "non_diegetic_music" in na
assert "N/A" in na
assert "Strict chronological timeline" not in na
assert "Do not anticipate" not in na
assert "Main visible action: Luluka (S1) dances." in na

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

assert mod.segments._context_frame_count(0.92, 124, 96) == 28
assert mod.segments._context_frame_count(2.0, 124, 96) == 45
assert mod.segments._context_frame_count(0.1, 124, 96) == 0
assert mod.segments._context_frame_count(2.0, 40, 96) == 28

previous_motion = s.MotionReferenceData(torch.ones(107, 16, 16, 3),
    {"waveform": torch.ones(1, 2, 142667), "sample_rate": 32000}, "仅动作", 4.0, 107 / 24, 4.0)
current_motion = s.MotionReferenceData(torch.full((107, 16, 16, 3), 2.0),
    {"waveform": torch.full((1, 2, 142667), 2.0), "sample_rate": 32000}, "仅动作", 4.0, 107 / 24, 4.0)
previous_clip = s.TimelineClipData("body", 0.0, 4.0, "walks forward", "", "", lang, "", "on-screen",
    None, "", previous_motion)
current_clip = s.TimelineClipData("body", 4.0, 8.0, "performs the dance", "", "", lang, "", "on-screen",
    None, "", current_motion)
segmented_track = s.TimelineTrackData("actor", actor, (previous_clip, current_clip))
segmented_timeline = s.TimelineData(group, style, env, s.TrackListData((segmented_track,)), 8.0)
job = s.GenerationJobData(segmented_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 2.0, "不输出")
compiled_segment, segment_references = mod.segments._compile_generation_segment(job, 1, 45)
segment_reference = segment_references[0]
assert compiled_segment.video_settings.length == 141
assert segment_reference.frames.shape[0] == 141
assert torch.all(segment_reference.frames[:45] == 1)
assert torch.all(segment_reference.frames[45:] == 2)
assert segment_reference.context_duration == 45 / 24
assert segment_reference.audio["waveform"].shape[-1] == 188000
assert torch.all(segment_reference.audio["waveform"][..., :60000] == 1)
assert torch.all(segment_reference.audio["waveform"][..., 60000:] == 2)
try:
    mod.segments._validate_motion_alignment((s.MotionReferenceData(
        segment_reference.frames[:-1], None, "仅动作", 4.0, 140 / 24),), 141)
    raise AssertionError("Motion reference length mismatch was accepted")
except ValueError as error:
    assert "140 帧" in str(error) and "141 帧" in str(error)
assert "1.62" not in compiled_segment.text and "5.62" not in compiled_segment.text
assert "Generate a continuous single shot." in compiled_segment.text
assert "The target video is a 4-second" not in compiled_segment.text
assert "<Subject 2> is Luluka (S1)'s motion sequence derived from <Video 1>" in compiled_segment.text
assert "Transfer <Subject 2> to Luluka (S1): continue the preceding motion" in compiled_segment.text
assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in compiled_segment.text

previous_without_reference = s.TimelineClipData("body", 0.0, 4.0, "walks forward", "", "", lang, "",
    "on-screen", None, "", None)
fallback_track = s.TimelineTrackData("actor", actor, (previous_without_reference, current_clip))
fallback_timeline = s.TimelineData(group, style, env, s.TrackListData((fallback_track,)), 8.0)
fallback_job = s.GenerationJobData(fallback_timeline, 0.4, "16:9", 0, "simple", 4, 1.0, "match", 2.0, "不输出")
generated_tail = torch.full((96, 16, 16, 3), 3.0)
generated_audio = {"waveform": torch.full((1, 2, 128000), 3.0), "sample_rate": 32000}
_, fallback_references = mod.segments._compile_generation_segment(fallback_job, 1, 45,
    generated_tail, generated_audio)
fallback_reference = fallback_references[0]
assert torch.all(fallback_reference.frames[:45] == 3)
assert torch.all(fallback_reference.frames[45:] == 2)
assert torch.all(fallback_reference.audio["waveform"][..., :60000] == 3)

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
locked = mod.segments._lock_context_prefix(latent, previous_images, previous_audio,
    VideoVAE(), AudioVAE(), 22, 64, 64)
video, audio = locked["samples"].unbind()
video_mask, audio_mask = locked["noise_mask"].unbind()
assert torch.all(video[:, :, :7] == 1)
assert torch.all(video_mask[:, :, :7] == 0) and torch.all(video_mask[:, :, 7:] == 1)
assert torch.all(audio[..., :37] == 1)
assert torch.all(audio_mask[..., :37] == 0) and torch.all(audio_mask[..., 37:] == 1)

try:
    mod.MiniMaxH3SegmentTrim.execute(torch.zeros(140, 16, 16, 3),
        {"waveform": torch.zeros(1, 2, 1000), "sample_rate": 32000}, 0, 4.0, 141)
    raise AssertionError("Decoded video length mismatch was accepted")
except ValueError as error:
    assert "140 帧" in str(error) and "141 帧" in str(error)

old_output = mod.checkpoints.folder_paths.get_output_directory()
with tempfile.TemporaryDirectory() as temp_output:
    mod.checkpoints.folder_paths.set_output_directory(temp_output)
    try:
        cache_name = "segment_001_0123456789abcdef01234567.mp4"
        components = mod.checkpoints.Types.VideoComponents(torch.zeros(5, 32, 32, 3),
            Fraction(24), {"waveform": torch.zeros(1, 2, 7000), "sample_rate": 32000})
        video = mod.checkpoints.InputImpl.VideoFromComponents(components)
        saved = mod.MiniMaxH3SegmentCheckpoint.execute(video, cache_name, f'["{cache_name}"]')[0]
        assert mod.checkpoints._cache_path(cache_name).is_file()
        loaded = mod.MiniMaxH3SegmentCheckpointLoad.execute(cache_name, f'["{cache_name}"]')[0]
        assert saved.get_dimensions() == loaded.get_dimensions() == (32, 32)
    finally:
        mod.checkpoints.folder_paths.set_output_directory(old_output)

empty_card = s.CharacterCardData("", "", "", None, "", "", "", "", "", "global")
empty_actor = s.ActorInstanceData(empty_card, "", "", "", "")
empty_group = s.CharacterGroupData((empty_actor,))
empty_style = s.StyleCardData("", "", "", "", None)
empty_env = s.EnvironmentInstanceData(s.EnvironmentCardData("", "", "", "", "", "", None), "", "", "", "")
empty_timeline = s.TimelineData(empty_group, empty_style, empty_env, s.TrackListData(()), 5.0)
empty_fl = mod.MiniMaxH3FinalPrompt.execute(empty_timeline, 0.98, "16:9", "FL", None, None, "",
    empty_sections="不输出")[0].text
assert empty_fl == ""

print("PASS")
