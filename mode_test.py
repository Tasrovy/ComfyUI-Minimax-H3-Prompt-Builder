import importlib.util
import pathlib
import sys

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

omit = mod.MiniMaxH3FinalPrompt.execute(timeline, 0.98, "16:9", "Ref", None, None, "",
    empty_sections="不输出")[0].text
na = mod.MiniMaxH3FinalPrompt.execute(timeline, 0.98, "16:9", "Ref", None, None, "",
    empty_sections="输出 N/A")[0].text

assert "overall_soundscape" not in omit
assert "non_diegetic_music" not in omit
assert "Continuous non-character controls" not in omit
assert "overall_soundscape" in na
assert "non_diegetic_music" in na
assert "Continuous non-character controls" in na
assert "N/A" in na

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
