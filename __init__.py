from typing_extensions import override

from comfy_api.latest import ComfyExtension

from .cards import (MiniMaxH3ActorInstance, MiniMaxH3Character, MiniMaxH3CharacterGroup,
    MiniMaxH3Environment, MiniMaxH3EnvironmentInstance, MiniMaxH3Language, MiniMaxH3Visual)
from .clips import (MiniMaxH3Action, MiniMaxH3ActionResult, MiniMaxH3ActorTrack, MiniMaxH3AudioAction, MiniMaxH3Camera,
    MiniMaxH3EnvironmentAction, MiniMaxH3EnvironmentTrack, MiniMaxH3LightingAction,
    MiniMaxH3MotionReference, MiniMaxH3SystemTrack, MiniMaxH3TrackList)
from .checkpoints import MiniMaxH3SegmentCheckpoint, MiniMaxH3SegmentCheckpointLoad
from .pipeline import (MiniMaxH3MultiSegmentGenerate, MiniMaxH3MultiSegmentSecondPass,
    MiniMaxH3SecondPassBatchAppend, MiniMaxH3SecondPassEntryPack, MiniMaxH3SecondPassLock,
    MiniMaxH3SecondPassResize, MiniMaxH3SecondPassUpscale, MiniMaxH3SegmentJoin,
    MiniMaxH3SegmentResultPrepare, MiniMaxH3SegmentSampler, MiniMaxH3SegmentTrim)
from .segments import (MiniMaxH3GenerationJob, MiniMaxH3PromptPreview, MiniMaxH3SegmentConditioning)
from .timeline import (MiniMaxH3FinalPrompt, MiniMaxH3PromptParser, MiniMaxH3Ref2VAAdapter, MiniMaxH3Timeline)


WEB_DIRECTORY = "./web"


class MiniMaxH3PromptBuilderExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [MiniMaxH3Character, MiniMaxH3ActorInstance, MiniMaxH3CharacterGroup, MiniMaxH3Language, MiniMaxH3Visual,
            MiniMaxH3Environment, MiniMaxH3EnvironmentInstance, MiniMaxH3Action, MiniMaxH3ActionResult,
            MiniMaxH3Camera, MiniMaxH3LightingAction,
            MiniMaxH3AudioAction, MiniMaxH3EnvironmentAction, MiniMaxH3ActorTrack, MiniMaxH3EnvironmentTrack,
            MiniMaxH3SystemTrack, MiniMaxH3TrackList, MiniMaxH3Timeline, MiniMaxH3FinalPrompt,
            MiniMaxH3PromptParser, MiniMaxH3Ref2VAAdapter, MiniMaxH3GenerationJob, MiniMaxH3MotionReference,
            MiniMaxH3PromptPreview, MiniMaxH3SegmentConditioning, MiniMaxH3SegmentTrim, MiniMaxH3SegmentJoin,
            MiniMaxH3SegmentResultPrepare, MiniMaxH3SegmentSampler,
            MiniMaxH3SecondPassResize, MiniMaxH3SecondPassEntryPack, MiniMaxH3SecondPassBatchAppend,
            MiniMaxH3SecondPassLock, MiniMaxH3SecondPassUpscale,
            MiniMaxH3SegmentCheckpoint, MiniMaxH3SegmentCheckpointLoad,
            MiniMaxH3MultiSegmentGenerate, MiniMaxH3MultiSegmentSecondPass]


async def comfy_entrypoint():
    return MiniMaxH3PromptBuilderExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
