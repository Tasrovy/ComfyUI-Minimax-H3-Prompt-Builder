# MiniMax H3 REF Timeline Prompt Builder

This custom-node package builds a single-shot MiniMax H3 Ref2VA prompt from reusable cards, instances, timed clips, and a unified track list.

## Data flow

```text
Character Card -> Actor Instance -> Character Group
                               \-> Actor Track --\
Environment Card -> Environment Instance -> Environment Track \
Camera / Lighting / Audio Clips -> System Tracks ----------------> Track List
Style Card + Character Group + Environment Instance + Track List -> Timeline
Timeline -> REF Timeline Compiler -> Prompt Parser -> Ref2VA Adapter
Timeline -> Generation Job -> Multi-Segment Generate -> VIDEO
                         \-> Final Prompt Preview
```

All collection nodes use native autogrow sockets. Connecting the last socket creates the next socket automatically.

## Cards and instances

- A Character Card stores identity, reference image, preservation rules, default initial state, and character-specific style.
- An Actor Instance inherits the card's initial state. Each non-empty instance field overrides the matching card field.
- A Character Group declares actor instances in connection order and assigns `S1`, `S2`, and so on automatically.
- An Environment Card and Environment Instance use the same inheritance and override rule.
- A Style Card supplies the global visual style. Each Character Card chooses whether its character-specific style or the global style wins when they conflict.

Character Card and Actor Instance state fields are predicates, so write `stands beneath a streetlight` instead of `L stands beneath a streetlight`. The compiler prefixes the current actor label automatically. This keeps cards reusable when a character is renamed or assigned a different `S` number.

Actor placeholders use the exact Character Group socket names. `{actor_0}` is the Actor Instance connected to `actor_0` and becomes its declared label such as `Lapsrk (S1)`; `{actor_1}` maps to `actor_1` and becomes `Name (S2)`, and so on. These placeholders can be used anywhere in the final prompt source, including camera, lighting, audio, and additional instructions. Compilation stops if an index exceeds the group size. Global style and environment fields should normally remain actor-independent.

## Timeline

Actor action clips have a selectable kind: `body`, `expression`, `gaze`, or `speech`. Language is an input of a `speech` action. Camera, lighting, audio, and environment nodes also produce the common `MINIMAX_H3_TIMELINE_CLIP` type.

Tracks all output `MINIMAX_H3_TIMELINE_TRACK`. The Track List therefore accepts a uniform array rather than separate actor, camera, lighting, audio, or environment inputs.

Time is edited in seconds and compared as half-open intervals `[start, end)`. A clip's process ends at `end`, while its non-empty `result` remains in effect until the next action of the same kind. Audio ends with its interval and may overlap other audio.

Compilation fails when non-audio clips with the same owner and kind overlap, when a clip exceeds the timeline duration, or when a track references an undeclared actor or another environment instance.

## Multi-segment generation

The native model loaders and patch nodes stay outside this package. Connect the patched `MODEL`, `CLIP`, video `VAE`, audio `VAE`, and `SAMPLER` directly to **MiniMax H3 Multi-Segment Generate**. Prompt, reference, size, seed, sampling, and segmentation settings are packed once by **MiniMax H3 Generation Job**.

The generator derives any number of generation segments from the timeline clips rather than from a target duration. Non-overlapping actor clips become sequential generation segments, while actor clips that overlap in time are generated together. If there are no actor clips, environment-change clips provide the segment structure. Camera, lighting, and audio tracks are supporting controls and are sliced into the affected segments without creating boundaries of their own.

During automatic multi-segment generation, the node displays the current segment number, internal stage, sampling step, and estimated overall percentage. Internal expanded-graph progress is mapped back to the visible Multi-Segment Generate node, so the existing compact workflow does not need to be split into manual chains.

Every segment uses Ref2VA. A later segment receives the preceding segment's tail as `<Video 1>` for continuity, followed by the motion-reference videos attached to its current action clips. The beginning of the new segment deliberately regenerates a short overlap; the join node finds the closest visual and motion match inside that overlap, removes the duplicate prefix, and then appends the remaining frames and audio. The expanded graph reuses ComfyUI's native guider, scheduler, sampler, VAE decoders, and video container. Its only output is a decoded `VIDEO` containing both frames and audio.

Use **MiniMax H3 Motion Reference** to trim a native `VIDEO` to the desired action, choose whether it controls only body motion or also camera, performance, or sound, and connect it to an Action clip. Ref2VA accepts at most three reference videos in one segment. Later segments reserve `<Video 1>` for continuity, so they can use at most two action-reference videos.

The Generation Job's reference-media size applies to both images and videos. `match` scales each reference down to the generated segment's pixel area, while `max` keeps the native high-quality reference canvas. Use `match` on limited VRAM because every reference-video token participates in every sampling step.

Connect the same Generation Job to **MiniMax H3 Final Prompt Preview** to inspect the exact complete Ref2VA prompt for every generated segment. The preview includes the original timeline range, regenerated overlap duration, continuity-video assignment, and action-reference video numbering used by the real generation path.

## Output

The standalone compiler produces a continuous single-shot `Ref` prompt with the six Ref2VA sections. Reference images are numbered automatically and returned in matching order by Prompt Parser. Multi-segment generation also compiles every segment as `Ref`; it no longer switches continuation segments to FL/I2VA.

Width and height are calculated from megapixels and aspect ratio, rounded to multiples of 32. Duration is converted at 24 fps and snapped upward to MiniMax H3's `17k+5` frame grid.
