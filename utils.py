import math
import re

import torch

import comfy.utils
from comfy_api.latest import io

from .schema import ASPECT_RATIOS, FPS, ReferenceImageData


def _text(value):
    return value.strip() if value else ""


def _sentence(value):
    value = _text(value)
    target = value[:-4].rstrip() if value.endswith("</d>") else value
    return value + "." if value and (not target or target[-1] not in ".!?。！？") else value


def _values(autogrow):
    return list(autogrow.values()) if autogrow else []


def _autogrow(input_type, name, prefix, minimum, maximum=100):
    return io.Autogrow.Input(name, optional=minimum == 0, template=io.Autogrow.TemplatePrefix(
        input=input_type.Input(prefix), prefix=f"{prefix}_", min=minimum, max=maximum,
    ))


def _reference(image, role, usage=""):
    return ReferenceImageData(0, image, role, _text(usage)) if image is not None else None


def _clip_has_content(clip):
    return bool(_text(clip.content))


def _same_image(left, right):
    return left is right or (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                             and left.shape == right.shape and torch.equal(left, right))


def _resolved(default, override):
    return _sentence(override) or _sentence(default)


_SUBJECT_PREFIXES = ("the character", "the woman", "the man", "the girl", "the boy", "she", "he")


def _strip_leading_subject(value, name=""):
    value = _text(value)
    if not value:
        return ""
    lowered = value.lower()
    prefixes = (name.lower(),) + _SUBJECT_PREFIXES if _text(name) else _SUBJECT_PREFIXES
    for prefix in prefixes:
        if lowered.startswith(prefix + " "):
            remainder = value[len(prefix):].lstrip()
            if remainder.lower().startswith("'s "):
                remainder = "is " + remainder[3:].lstrip()
            return _text(remainder)
        if lowered.startswith(prefix + "'s "):
            return _text("is " + value[len(prefix) + 3:].lstrip())
    return value


def _card_description(value, name=""):
    value = _sentence(_strip_leading_subject(value, name))
    if value.lower().startswith("is "):
        value = _sentence(value[3:])
    return value


def _lower_first(value):
    value = _text(value)
    return value[:1].lower() + value[1:] if value else ""


def _actor_state(label, default, override):
    state = _text(override) or _text(default)
    if not state:
        return ""
    name = label.rsplit(" (S", 1)[0] if " (S" in label else ""
    state = _strip_leading_subject(state, name)
    if not state:
        return ""
    return _sentence(f"{label} {state}")


def _bind_actor_tokens(value, actor_labels):
    value = _text(value)
    if not value:
        return ""

    def replace_actor(match):
        socket_suffix = match.group(1)
        if not re.fullmatch(r"_\d+", socket_suffix):
            raise ValueError(f"Invalid actor placeholder {match.group(0)}; use {{actor_0}}, {{actor_1}}, and so on")
        index = int(socket_suffix[1:])
        if index >= len(actor_labels):
            raise ValueError(f"{match.group(0)} exceeds the Character Group, which contains {len(actor_labels)} actor(s)")
        return actor_labels[index]

    return re.sub(r"\{actor([^}]*)\}", replace_actor, value)


def _video_size(megapixels, aspect_ratio):
    ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
    scale = math.sqrt(megapixels * 1024 * 1024 / (ratio_width * ratio_height))
    return round(ratio_width * scale / 32) * 32, round(ratio_height * scale / 32) * 32


def _video_length(duration):
    frames = max(5, math.ceil(duration * FPS))
    remainder = (frames - 5) % 17
    return frames if not remainder else frames + 17 - remainder


def _match_reference_video(frames, width, height):
    source_height, source_width = frames.shape[1:3]
    scale = min(1.0, math.sqrt((width * height) / (source_width * source_height)))
    target_width = max(32, round(source_width * scale / 32) * 32)
    target_height = max(32, round(source_height * scale / 32) * 32)
    if (target_height, target_width) == (source_height, source_width):
        return frames
    samples = frames[..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, target_width, target_height, "lanczos", "disabled").movedim(1, -1)


def _time(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")
