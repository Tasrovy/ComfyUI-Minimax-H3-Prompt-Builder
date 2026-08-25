import json
import os
import re
import tempfile
import uuid

from aiohttp import web
import folder_paths
from server import PromptServer


FORMAT = "minimax-h3-resource-library"
INDEX_FORMAT = "minimax-h3-resource-index"
CARD_FORMAT = "minimax-h3-resource-card"
VERSION = 1
RESOURCE_KINDS = ("characters", "environments", "styles")
CARD_KINDS = {"characters": "character", "environments": "environment", "styles": "style"}
KIND_DIRECTORIES = {value: key for key, value in CARD_KINDS.items()}
ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,96}$")
MAX_RESOURCES = 1000
MAX_TEXT = 20000
MAX_FILE_BYTES = 2 * 1024 * 1024


def _default_library():
    return {
        "format": FORMAT,
        "version": VERSION,
        "library_id": "main-library",
        "name": "MiniMax H3 主资源库",
        "revision": 0,
        "characters": [],
        "environments": [],
        "styles": [],
        "extensions": {},
        "extension_cards": [],
    }


def _library_path():
    directory = os.path.join(folder_paths.get_user_directory(), "minimax_h3", "resources")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "library.json")


def _resource_directory():
    return os.path.dirname(_library_path())


def _json_object(value, label):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是对象")
    return value


def _text(value, label, required=False):
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是字符串")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{label}不能为空")
    if len(value) > MAX_TEXT:
        raise ValueError(f"{label}过长")
    return value


def _resource_id(value):
    value = _text(value, "资源 ID")
    if not value:
        return str(uuid.uuid4())
    if not ID_PATTERN.fullmatch(value):
        raise ValueError("资源 ID 只能包含字母、数字、点、下划线和连字符")
    return value


def _tags(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError("标签必须是不超过 64 项的数组")
    result = []
    for item in value:
        item = _text(item, "标签")
        if item:
            result.append(item[:100])
    return result


def _reference_image(value):
    if value in (None, {}):
        return None
    if not isinstance(value, dict):
        raise ValueError("参考图信息必须是对象")
    folder_type = value.get("type", "input")
    if folder_type != "input":
        raise ValueError("资源库参考图必须位于 ComfyUI input 目录")
    filename = _text(value.get("filename"), "参考图文件名", True)
    subfolder = _text(value.get("subfolder"), "参考图子目录").replace("\\", "/")
    if (os.path.basename(filename) != filename or subfolder.startswith("/") or ":" in subfolder or
            ".." in subfolder.split("/")):
        raise ValueError("参考图路径不安全")
    return {"type": "input", "subfolder": subfolder.strip("/"), "filename": filename}


def _character(value):
    if not isinstance(value, dict):
        raise ValueError("人物资源必须是对象")
    card = value.get("card") or {}
    defaults = value.get("instance_defaults") or {}
    if not isinstance(card, dict) or not isinstance(defaults, dict):
        raise ValueError("人物卡内容格式错误")
    name = _text(card.get("name") or value.get("display_name"), "人物名称", True)
    return {
        "id": _resource_id(value.get("id")),
        "revision": max(1, int(value.get("revision", 1))),
        "display_name": _text(value.get("display_name") or name, "显示名称", True),
        "tags": _tags(value.get("tags")),
        "reference_image": _reference_image(value.get("reference_image")),
        "extensions": _json_object(value.get("extensions"), "人物扩展数据"),
        "card": {
            "name": name,
            "description": _text(card.get("description"), "人物外观"),
            "style_priority": card.get("style_priority") if card.get("style_priority") in ("character", "global") else "global",
            "character_style": _text(card.get("character_style"), "人物风格"),
        },
        "instance_defaults": {
            "position_override": _text(defaults.get("position_override"), "默认位置"),
            "pose_override": _text(defaults.get("pose_override"), "默认姿态"),
            "emotion_override": _text(defaults.get("emotion_override"), "默认表情"),
            "appearance_override": _text(defaults.get("appearance_override"), "默认附加状态"),
        },
    }


def _environment(value):
    if not isinstance(value, dict):
        raise ValueError("环境资源必须是对象")
    card = value.get("card") or {}
    defaults = value.get("instance_defaults") or {}
    if not isinstance(card, dict) or not isinstance(defaults, dict):
        raise ValueError("环境卡内容格式错误")
    name = _text(card.get("name") or value.get("display_name"), "环境名称", True)
    return {
        "id": _resource_id(value.get("id")),
        "revision": max(1, int(value.get("revision", 1))),
        "display_name": _text(value.get("display_name") or name, "显示名称", True),
        "tags": _tags(value.get("tags")),
        "reference_image": _reference_image(value.get("reference_image")),
        "extensions": _json_object(value.get("extensions"), "环境扩展数据"),
        "card": {
            "name": name,
            "location": _text(card.get("location"), "空间与地点外观"),
            "default_background": _text(card.get("default_background"), "固定背景与陈设"),
        },
        "instance_defaults": {
            "location_override": _text(defaults.get("location_override"), "当前地点变体"),
            "time_weather_override": _text(defaults.get("time_weather_override"), "当前时间与天气"),
            "background_override": _text(defaults.get("background_override"), "当前背景变体"),
            "atmosphere_override": _text(defaults.get("atmosphere_override"), "当前环境氛围"),
        },
    }


def _style(value):
    if not isinstance(value, dict):
        raise ValueError("风格资源必须是对象")
    card = value.get("card") or {}
    if not isinstance(card, dict):
        raise ValueError("风格卡内容格式错误")
    display_name = _text(value.get("display_name"), "风格名称", True)
    return {
        "id": _resource_id(value.get("id")),
        "revision": max(1, int(value.get("revision", 1))),
        "display_name": display_name,
        "tags": _tags(value.get("tags")),
        "reference_image": _reference_image(value.get("reference_image")),
        "extensions": _json_object(value.get("extensions"), "风格扩展数据"),
        "card": {
            "style": _text(card.get("style"), "视觉风格"),
            "rendering": _text(card.get("rendering"), "渲染表现"),
            "color_palette": _text(card.get("color_palette"), "色彩方案"),
            "texture": _text(card.get("texture"), "画面质感"),
            "reference_usage": _text(card.get("reference_usage"), "参考图用途"),
        },
    }


def _card_kind(value):
    value = _text(value, "卡片类型", True)
    if not ID_PATTERN.fullmatch(value):
        raise ValueError("卡片类型只能包含字母、数字、点、下划线和连字符")
    return value


def _extension_card(value):
    if not isinstance(value, dict):
        raise ValueError("扩展卡片必须是对象")
    kind = _card_kind(value.get("kind"))
    if kind in KIND_DIRECTORIES:
        raise ValueError(f"核心卡片类型不能放入扩展卡片：{kind}")
    if value.get("format", CARD_FORMAT) != CARD_FORMAT or int(value.get("version", VERSION)) != VERSION:
        raise ValueError("不支持的扩展卡片格式或版本")
    data = _json_object(value.get("data"), "扩展卡片数据")
    extensions = _json_object(value.get("extensions"), "扩展卡片扩展数据")
    return {
        "format": CARD_FORMAT,
        "version": VERSION,
        "kind": kind,
        "id": _resource_id(value.get("id")),
        "revision": max(1, int(value.get("revision", 1))),
        "display_name": _text(value.get("display_name"), "扩展卡片名称", True),
        "tags": _tags(value.get("tags")),
        "reference_image": _reference_image(value.get("reference_image")),
        "data": data,
        "extensions": extensions,
    }


def normalize_library(value):
    if not isinstance(value, dict):
        raise ValueError("资源库必须是 JSON 对象")
    if value.get("format", FORMAT) != FORMAT or int(value.get("version", VERSION)) != VERSION:
        raise ValueError("不支持的资源库格式或版本")
    normalizers = {"characters": _character, "environments": _environment, "styles": _style}
    result = _default_library()
    result["library_id"] = _resource_id(value.get("library_id") or "main-library")
    result["name"] = _text(value.get("name") or result["name"], "资源库名称", True)
    result["revision"] = max(0, int(value.get("revision", 0)))
    result["extensions"] = _json_object(value.get("extensions"), "资源库扩展数据")
    seen = set()
    count = 0
    for kind in RESOURCE_KINDS:
        items = value.get(kind, [])
        if not isinstance(items, list):
            raise ValueError(f"{kind}必须是数组")
        result[kind] = []
        for item in items:
            normalized = normalizers[kind](item)
            if normalized["id"] in seen:
                raise ValueError(f"资源 ID 重复：{normalized['id']}")
            seen.add(normalized["id"])
            result[kind].append(normalized)
            count += 1
    if count > MAX_RESOURCES:
        raise ValueError(f"资源数量不能超过 {MAX_RESOURCES}")
    extension_cards = value.get("extension_cards", [])
    if not isinstance(extension_cards, list):
        raise ValueError("extension_cards必须是数组")
    for item in extension_cards:
        normalized = _extension_card(item)
        if normalized["id"] in seen:
            raise ValueError(f"资源 ID 重复：{normalized['id']}")
        seen.add(normalized["id"])
        result["extension_cards"].append(normalized)
        count += 1
    if count > MAX_RESOURCES:
        raise ValueError(f"资源数量不能超过 {MAX_RESOURCES}")
    return result


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + "-", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_if_changed(path, value):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            if json.load(file) == value:
                return
    _write_json(path, value)


def _card_path(kind, resource_id):
    kind = _card_kind(kind)
    resource_id = _resource_id(resource_id)
    return os.path.join(_resource_directory(), "cards", kind, resource_id + ".json")


def _card_document(kind, resource):
    data = {"card": resource["card"]}
    if "instance_defaults" in resource:
        data["instance_defaults"] = resource["instance_defaults"]
    return {
        "format": CARD_FORMAT,
        "version": VERSION,
        "kind": kind,
        "id": resource["id"],
        "revision": resource["revision"],
        "display_name": resource["display_name"],
        "tags": resource["tags"],
        "reference_image": resource["reference_image"],
        "data": data,
        "extensions": resource.get("extensions", {}),
    }


def _index_document(library):
    return {
        "format": INDEX_FORMAT,
        "version": VERSION,
        "library_id": library["library_id"],
        "name": library["name"],
        "revision": library["revision"],
        "extensions": library.get("extensions", {}),
    }


def _card_files():
    cards_directory = os.path.join(_resource_directory(), "cards")
    if not os.path.isdir(cards_directory):
        return []
    result = []
    for kind_entry in os.scandir(cards_directory):
        if not kind_entry.is_dir(follow_symlinks=False) or not ID_PATTERN.fullmatch(kind_entry.name):
            continue
        for card_entry in os.scandir(kind_entry.path):
            if card_entry.is_file(follow_symlinks=False) and card_entry.name.endswith(".json"):
                result.append((kind_entry.name, card_entry.path))
    return sorted(result)


def _load_cards(library):
    normalizers = {CARD_KINDS[key]: value for key, value in {
        "characters": _character, "environments": _environment, "styles": _style,
    }.items()}
    seen = set()
    for directory_kind, path in _card_files():
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise ValueError(f"卡片文件不能超过 2 MB：{os.path.basename(path)}")
        with open(path, "r", encoding="utf-8") as file:
            document = json.load(file)
        if not isinstance(document, dict) or document.get("format") != CARD_FORMAT or int(document.get("version", 0)) != VERSION:
            raise ValueError(f"卡片文件格式错误：{os.path.basename(path)}")
        kind = _card_kind(document.get("kind"))
        resource_id = _resource_id(document.get("id"))
        if kind != directory_kind or os.path.basename(path) != resource_id + ".json":
            raise ValueError(f"卡片文件路径与卡片身份不一致：{os.path.basename(path)}")
        if resource_id in seen:
            raise ValueError(f"资源 ID 重复：{resource_id}")
        seen.add(resource_id)
        if len(seen) > MAX_RESOURCES:
            raise ValueError(f"资源数量不能超过 {MAX_RESOURCES}")
        if kind in normalizers:
            data = _json_object(document.get("data"), "卡片数据")
            resource = {key: document.get(key) for key in ("id", "revision", "display_name", "tags", "reference_image", "extensions")}
            resource.update(data)
            library[KIND_DIRECTORIES[kind]].append(normalizers[kind](resource))
        else:
            library["extension_cards"].append(_extension_card(document))
    return library


def load_library():
    path = _library_path()
    if not os.path.exists(path):
        return _load_cards(_default_library())
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("资源索引必须是 JSON 对象")
    if value.get("format") != INDEX_FORMAT:
        return normalize_library(value)
    if int(value.get("version", 0)) != VERSION:
        raise ValueError("不支持的资源索引版本")
    library = _default_library()
    library["library_id"] = _resource_id(value.get("library_id") or "main-library")
    library["name"] = _text(value.get("name") or library["name"], "资源库名称", True)
    library["revision"] = max(0, int(value.get("revision", 0)))
    library["extensions"] = _json_object(value.get("extensions"), "资源库扩展数据")
    return _load_cards(library)


def save_library(value, expected_revision):
    current = load_library()
    if expected_revision is not None and int(expected_revision) != current["revision"]:
        raise RuntimeError("资源库已被其他页面修改，请刷新后重试")
    normalized = normalize_library(value)
    normalized["revision"] = current["revision"] + 1
    active_paths = set()
    for directory, kind in CARD_KINDS.items():
        for resource in normalized[directory]:
            path = _card_path(kind, resource["id"])
            _write_json_if_changed(path, _card_document(kind, resource))
            active_paths.add(os.path.normcase(path))
    for document in normalized["extension_cards"]:
        path = _card_path(document["kind"], document["id"])
        _write_json_if_changed(path, document)
        active_paths.add(os.path.normcase(path))
    for _, path in _card_files():
        if os.path.normcase(path) not in active_paths:
            os.unlink(path)
    _write_json(_library_path(), _index_document(normalized))
    return normalized


async def get_resources(request):
    try:
        return web.json_response({"success": True, "library": load_library()})
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return web.json_response({"success": False, "error": str(error)}, status=500)


async def put_resources(request):
    if request.content_length is not None and request.content_length > 2 * 1024 * 1024:
        return web.json_response({"success": False, "error": "资源库文件不能超过 2 MB"}, status=413)
    try:
        raw = await request.read()
        if len(raw) > 2 * 1024 * 1024:
            return web.json_response({"success": False, "error": "资源库文件不能超过 2 MB"}, status=413)
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("请求内容必须是 JSON 对象")
        library = save_library(body.get("library"), body.get("expected_revision"))
        return web.json_response({"success": True, "library": library})
    except RuntimeError as error:
        return web.json_response({"success": False, "error": str(error)}, status=409)
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        return web.json_response({"success": False, "error": str(error)}, status=400)


def register_routes():
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    server.routes.get("/minimax-h3/resources")(get_resources)
    server.routes.put("/minimax-h3/resources")(put_resources)


register_routes()
