import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TYPES = {
    timeline: "MiniMaxH3Timeline",
    trackList: "MiniMaxH3TrackList",
    actorTrack: "MiniMaxH3ActorTrack",
    environmentTrack: "MiniMaxH3EnvironmentTrack",
    systemTrack: "MiniMaxH3SystemTrack",
    action: "MiniMaxH3Action",
    actionResult: "MiniMaxH3ActionResult",
    camera: "MiniMaxH3Camera",
    lighting: "MiniMaxH3LightingAction",
    audio: "MiniMaxH3AudioAction",
    environmentAction: "MiniMaxH3EnvironmentAction",
    character: "MiniMaxH3Character",
    actor: "MiniMaxH3ActorInstance",
    visual: "MiniMaxH3Visual",
    environment: "MiniMaxH3Environment",
    environmentInstance: "MiniMaxH3EnvironmentInstance",
    characterGroup: "MiniMaxH3CharacterGroup",
    language: "MiniMaxH3Language",
    motionReference: "MiniMaxH3MotionReference",
    videoPerson: "MiniMaxH3ReferenceVideoPerson",
    loadImage: "LoadImage",
    loadVideo: "LoadVideo",
};

const FIELD_SETS = {
    [TYPES.action]: [
        ["action_type", "动作种类"], ["start_time", "开始时间（秒）"], ["end_time", "结束时间（秒）"],
        ["use_previous_context", "继承前一动作片段的段间引导"],
        ["audio_only_context", "仅段间音频引导"],
        ["content", "动作或台词内容", "textarea"], ["speech_type", "说话类型"],
        ["quality", "动作质量", "textarea"], ["result", "结束状态", "textarea"], ["delivery", "说话方式", "textarea"],
    ],
    [TYPES.camera]: [
        ["start_time", "开始时间（秒）"], ["end_time", "结束时间（秒）"],
        ["framing_and_angle", "景别与角度", "textarea"], ["movement", "摄像机运动", "textarea"],
        ["focus", "对焦与景深", "textarea"], ["result", "结束状态", "textarea"],
    ],
    [TYPES.lighting]: [
        ["start_time", "开始时间（秒）"], ["end_time", "结束时间（秒）"],
        ["lighting", "灯光描述", "textarea"], ["transition", "灯光变化", "textarea"], ["result", "结束状态", "textarea"],
    ],
    [TYPES.audio]: [
        ["audio_type", "音频种类"], ["start_time", "开始时间（秒）"], ["end_time", "结束时间（秒）"],
        ["description", "声音描述", "textarea"], ["volume_and_space", "音量与空间", "textarea"], ["fade", "淡入淡出", "textarea"],
    ],
    [TYPES.environmentAction]: [
        ["start_time", "开始时间（秒）"], ["end_time", "结束时间（秒）"],
        ["change", "环境变化", "textarea"], ["quality", "变化质量", "textarea"], ["result", "结束状态", "textarea"],
    ],
    [TYPES.character]: [
        ["name", "人物名称"], ["description", "人物外观", "textarea"], ["style_priority", "风格优先级"],
        ["character_style", "人物风格", "textarea"],
    ],
    [TYPES.actor]: [
        ["actor_id", "人物实例宏"],
        ["position_override", "无动作时位置", "textarea"], ["pose_override", "无动作时姿态", "textarea"],
        ["emotion_override", "无动作时表情", "textarea"], ["appearance_override", "无动作时附加状态", "textarea"],
    ],
    [TYPES.visual]: [
        ["style", "视觉风格", "textarea"], ["rendering", "渲染表现", "textarea"],
        ["color_palette", "色彩方案", "textarea"], ["texture", "画面质感", "textarea"],
        ["reference_usage", "参考图用途", "textarea"],
    ],
    [TYPES.environment]: [
        ["name", "环境名称"], ["location", "空间与地点外观", "textarea"],
        ["default_background", "固定背景与陈设", "textarea"],
    ],
    [TYPES.environmentInstance]: [
        ["location_override", "当前地点变体", "textarea"], ["time_weather_override", "当前时间与天气", "textarea"],
        ["background_override", "当前背景变体", "textarea"], ["atmosphere_override", "当前环境氛围", "textarea"],
    ],
    [TYPES.videoPerson]: [
        ["person_id", "源人物编号"], ["person_description", "源人物识别描述", "textarea"],
    ],
};

const KIND_NAMES = {
    body: "肢体", expression: "表情", gaze: "视线", speech: "对话",
    camera: "镜头", lighting: "灯光", audio: "音频", environment: "环境",
};

const CLIP_TYPES = new Set([TYPES.action, TYPES.camera, TYPES.lighting, TYPES.audio, TYPES.environmentAction]);
const PACKAGE_NODE_TYPES = new Set([
    ...Object.values(TYPES).filter((type) => type.startsWith("MiniMaxH3")),
    "MiniMaxH3FinalPrompt", "MiniMaxH3PromptParser", "MiniMaxH3Ref2VAAdapter",
    "MiniMaxH3GenerationJob", "MiniMaxH3PromptPreview", "MiniMaxH3SegmentConditioning",
    "MiniMaxH3SegmentTrim", "MiniMaxH3SegmentJoin", "MiniMaxH3SegmentResultPrepare",
    "MiniMaxH3SegmentSampler", "MiniMaxH3SecondPassResize", "MiniMaxH3SecondPassEntryPack",
    "MiniMaxH3SecondPassBatchAppend", "MiniMaxH3SecondPassBatchParser", "MiniMaxH3SecondPassLock",
    "MiniMaxH3SecondPassUpscale", "MiniMaxH3SegmentCheckpoint", "MiniMaxH3SegmentCheckpointLoad",
    "MiniMaxH3MultiSegmentGenerate", "MiniMaxH3MultiSegmentSecondPass",
    "MiniMaxH3MultiSegmentLatentSecondPass",
]);
const RESOURCE_NAMES = { characters: "人物", environments: "环境", styles: "风格" };
const PROJECT_FORMAT = "minimax-h3-director-project";
const PROJECT_VERSION = 1;
const PROJECT_CLIP_TYPES = {
    action: TYPES.action,
    camera: TYPES.camera,
    lighting: TYPES.lighting,
    audio: TYPES.audio,
    environment: TYPES.environmentAction,
};
const TRACK_KIND_NAMES = { body: "人物·肢体", expression: "人物·表情", gaze: "人物·视线", environment: "环境", camera: "镜头", lighting: "灯光", audio: "音频" };
const REFERENCE_SLOTS = {
    [TYPES.action]: { input: "motion_reference", output: 0, label: "人物表演" },
    [TYPES.camera]: { input: "camera_reference", output: 1, label: "镜头" },
    [TYPES.lighting]: { input: "lighting_reference", output: 2, label: "灯光" },
    [TYPES.audio]: { input: "audio_reference", output: 3, label: "音频" },
    [TYPES.environmentAction]: { input: "environment_reference", output: 4, label: "环境" },
};
const RESOURCE_FORMS = {
    characters: [
        ["display_name", "资源名称"], ["tags", "标签（逗号分隔）"], ["card.name", "人物名称"],
        ["card.description", "人物固定外观", "textarea"], ["card.style_priority", "风格优先级", "select", ["global", "character"]],
        ["card.character_style", "人物专属风格", "textarea"],
        ["instance_defaults.position_override", "默认静态位置", "textarea"], ["instance_defaults.pose_override", "默认静态姿态", "textarea"],
        ["instance_defaults.emotion_override", "默认静态表情", "textarea"], ["instance_defaults.appearance_override", "默认附加状态", "textarea"],
    ],
    environments: [
        ["display_name", "资源名称"], ["tags", "标签（逗号分隔）"], ["card.name", "环境名称"],
        ["card.location", "空间与地点外观", "textarea"], ["card.default_background", "固定背景与陈设", "textarea"],
        ["instance_defaults.location_override", "当前地点变体", "textarea"], ["instance_defaults.time_weather_override", "当前时间与天气", "textarea"],
        ["instance_defaults.background_override", "当前背景变体", "textarea"], ["instance_defaults.atmosphere_override", "当前环境氛围", "textarea"],
    ],
    styles: [
        ["display_name", "资源名称"], ["tags", "标签（逗号分隔）"], ["card.style", "视觉风格", "textarea"],
        ["card.rendering", "渲染表现", "textarea"], ["card.color_palette", "色彩方案", "textarea"],
        ["card.texture", "画面质感", "textarea"], ["card.reference_usage", "参考图用途", "textarea"],
    ],
};
const $ = (selector, root = document) => root.querySelector(selector);

function nodeType(node) {
    return node?.comfyClass || node?.type || "";
}

function widget(node, name) {
    return node?.widgets?.find((item) => item.name === name);
}

function widgetValue(node, name, fallback = "") {
    const item = widget(node, name);
    return item?.value ?? fallback;
}

function migrateActionContextWidget(node) {
    if (nodeType(node) !== TYPES.action) return false;
    const context = widget(node, "use_previous_context");
    if (!context || typeof context.value === "boolean") return false;
    if (context.value === "true" || context.value === "false") {
        context.value = context.value === "true";
        node._minimaxH3ContextMigrated = true;
        return true;
    }
    const quality = widget(node, "quality");
    const result = widget(node, "result");
    const delivery = widget(node, "delivery");
    const oldQuality = context.value;
    const oldResult = quality?.value ?? "";
    const oldDelivery = result?.value ?? "";
    context.value = true;
    if (quality) quality.value = oldQuality;
    if (result) result.value = oldResult;
    if (delivery) delivery.value = oldDelivery;
    node._minimaxH3ContextMigrated = true;
    return true;
}

function linkedNode(node, inputName) {
    const input = node?.inputs?.find((item) => item.name === inputName);
    if (input?.link == null) return null;
    const link = app.graph?.links?.[input.link];
    return link ? app.graph.getNodeById(link.origin_id) : null;
}

function numberedInputs(node, prefixes) {
    const names = Array.isArray(prefixes) ? prefixes : [prefixes];
    return (node?.inputs || [])
        .filter((input) => {
            const leaf = input.name.split(".").at(-1);
            return names.some((prefix) => input.name === prefix || input.name.startsWith(`${prefix}.`) || leaf === prefix || leaf.startsWith(`${prefix}_`));
        })
        .sort((a, b) => {
            const ai = Number(a.name.match(/(\d+)$/)?.[1] ?? 0);
            const bi = Number(b.name.match(/(\d+)$/)?.[1] ?? 0);
            return ai - bi;
        })
        .map((input) => linkedNode(node, input.name))
        .filter(Boolean);
}

function setWidgetValue(node, name, value) {
    const input = node?.inputs?.find((item) => item.name === name);
    if (input?.link != null) return false;
    const item = widget(node, name);
    if (!item) return false;
    const previous = item.value;
    item.value = value;
    item.callback?.(value, app.canvas, node, item);
    node.onWidgetChanged?.(name, value, previous, item);
    app.graph?.setDirtyCanvas(true, true);
    if (editor) editor.markWorkflowChanged();
    else app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
    return true;
}

function addWidgetOption(node, name, value) {
    const item = widget(node, name);
    const values = item?.options?.values;
    if (!item || !Array.isArray(values) || values.includes(value)) return;
    values.push(value);
}

function setImageLoaderValue(node, value) {
    addWidgetOption(node, "image", value);
    return setWidgetValue(node, "image", value);
}

function escapeHtml(value) {
    const element = document.createElement("div");
    element.textContent = String(value ?? "");
    return element.innerHTML;
}

function escapeAttribute(value) {
    return escapeHtml(value).replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function shortText(value, limit = 48) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function pathValue(object, path, fallback = "") {
    return path.split(".").reduce((value, key) => value?.[key], object) ?? fallback;
}

function setPathValue(object, path, value) {
    const parts = path.split(".");
    const last = parts.pop();
    const target = parts.reduce((current, key) => current[key] ||= {}, object);
    target[last] = value;
}

function slotIndex(node, name, output = false) {
    const slots = output ? node?.outputs : node?.inputs;
    return (slots || []).findIndex((slot) => slot.name === name || slot.label === name || slot.name?.split(".").at(-1) === name);
}

function connectNodes(origin, outputName, target, inputName) {
    const output = slotIndex(origin, outputName, true);
    const input = slotIndex(target, inputName);
    if (output < 0 || input < 0) return false;
    origin.connect(output, target, input);
    return true;
}

function outputTargets(node) {
    return (node?.outputs || []).flatMap((output) => output.links || [])
        .map((linkId) => app.graph?.links?.[linkId]?.target_id)
        .filter((id) => id != null);
}

function isManagedMediaNode(node) {
    if (node?.properties?.minimax_h3_managed_media || node?.properties?.minimax_h3_reference_upload
        || node?.properties?.minimax_h3_reference_asset) return true;
    if (nodeType(node) === TYPES.loadImage) {
        return String(widgetValue(node, "image", "")).replaceAll("\\", "/").startsWith("minimax_h3/");
    }
    if (nodeType(node) === TYPES.loadVideo) {
        return String(widgetValue(node, "file", "")).replaceAll("\\", "/").startsWith("minimax_h3/");
    }
    return false;
}

function isCleanupCandidate(node) {
    return PACKAGE_NODE_TYPES.has(nodeType(node)) || isManagedMediaNode(node);
}

function unusedGraphNodes(root) {
    if (!root) return [];
    const used = new Set();
    const pending = [root];
    while (pending.length) {
        const node = pending.pop();
        if (!node || used.has(node.id)) continue;
        used.add(node.id);
        for (const input of node.inputs || []) {
            if (input.link == null) continue;
            const origin = app.graph?.getNodeById(app.graph.links?.[input.link]?.origin_id);
            if (origin && !used.has(origin.id)) pending.push(origin);
        }
        for (const output of node.outputs || []) {
            for (const linkId of output.links || []) {
                const target = app.graph?.getNodeById(app.graph.links?.[linkId]?.target_id);
                if (target && !used.has(target.id)) pending.push(target);
            }
        }
    }
    return (app.graph?._nodes || []).filter((node) => !used.has(node.id) && isCleanupCandidate(node));
}

function upstreamGraphNodes(root) {
    const upstream = new Map();
    const pending = [root];
    while (pending.length) {
        const node = pending.pop();
        for (const input of node?.inputs || []) {
            if (input.link == null) continue;
            const origin = app.graph?.getNodeById(app.graph.links?.[input.link]?.origin_id);
            if (!origin || upstream.has(origin.id)) continue;
            upstream.set(origin.id, origin);
            pending.push(origin);
        }
    }
    return [...upstream.values()];
}

function abandonedUpstreamNodes(nodes) {
    const candidates = new Set(nodes.map((node) => node.id));
    const abandoned = new Set();
    let changed = true;
    while (changed) {
        changed = false;
        for (const node of nodes) {
            if (abandoned.has(node.id)) continue;
            const targets = outputTargets(node);
            if (targets.every((id) => abandoned.has(id))) {
                abandoned.add(node.id);
                changed = true;
            } else if (targets.some((id) => !candidates.has(id))) {
                continue;
            }
        }
    }
    return nodes.filter((node) => abandoned.has(node.id));
}

function freeAutogrowInput(node, prefix) {
    return (node?.inputs || []).findIndex((input) => {
        const leaf = input.name.split(".").at(-1);
        return leaf.startsWith(`${prefix}_`) && input.link == null;
    });
}

function clipKind(node) {
    switch (nodeType(node)) {
        case TYPES.action: return widgetValue(node, "action_type", "body");
        case TYPES.camera: return "camera";
        case TYPES.lighting: return "lighting";
        case TYPES.audio: return "audio";
        case TYPES.environmentAction: return "environment";
        default: return "body";
    }
}

function clipText(node) {
    const kind = clipKind(node);
    const field = kind === "camera" ? "movement" : kind === "lighting" ? "lighting" : kind === "audio" ? "description" : kind === "environment" ? "change" : "content";
    return shortText(widgetValue(node, field)) || KIND_NAMES[kind] || "片段";
}

function unwrapClip(node) {
    if (nodeType(node) !== TYPES.actionResult) return { node, resultNode: null };
    return { node: linkedNode(node, "clip"), resultNode: node };
}

function findCardForActor(actor) {
    return linkedNode(actor, "character_card");
}

function findCardForEnvironment(environment) {
    return linkedNode(environment, "environment_card");
}

function actorName(actor, fallback) {
    const card = findCardForActor(actor);
    return widgetValue(card, "name", fallback) || fallback;
}

function environmentName(environment, fallback) {
    const card = findCardForEnvironment(environment);
    return widgetValue(card, "name", fallback) || fallback;
}

function collectTimeline(timeline) {
    const duration = Math.max(0.21, Number(widgetValue(timeline, "duration_seconds", 5)) || 5);
    const trackList = linkedNode(timeline, "tracks");
    const trackNodes = nodeType(trackList) === TYPES.trackList ? numberedInputs(trackList, ["tracks", "track"]) : [];
    const tracks = trackNodes.map((trackNode, index) => {
        const type = nodeType(trackNode);
        const owner = type === TYPES.actorTrack ? linkedNode(trackNode, "actor") : type === TYPES.environmentTrack ? linkedNode(trackNode, "environment") : null;
        const fallback = type === TYPES.actorTrack ? `人物 ${index + 1}` : type === TYPES.environmentTrack ? "环境" : "系统";
        let label = type === TYPES.actorTrack ? actorName(owner, fallback) : type === TYPES.environmentTrack ? environmentName(owner, fallback) : fallback;
        const clips = numberedInputs(trackNode, ["clips", "clip"]).map((source, clipIndex) => {
            const unwrapped = unwrapClip(source);
            if (!unwrapped.node || !CLIP_TYPES.has(nodeType(unwrapped.node))) return null;
            return {
                node: unwrapped.node,
                resultNode: unwrapped.resultNode,
                trackNode,
                trackIndex: index,
                clipIndex,
                kind: clipKind(unwrapped.node),
                start: Number(widgetValue(unwrapped.node, "start_time", 0)) || 0,
                end: Number(widgetValue(unwrapped.node, "end_time", 1)) || 1,
            };
        }).filter(Boolean);
        if (type === TYPES.systemTrack && clips.length) label = `${KIND_NAMES[clips[0].kind] || "系统"}轨道`;
        return { node: trackNode, owner, label, type, clips };
    });
    return {
        timeline,
        duration,
        promptLanguage: widgetValue(timeline, "prompt_language", "英文") || "英文",
        tracks,
        characterGroup: linkedNode(timeline, "character_group"),
        style: linkedNode(timeline, "style_card"),
        environment: linkedNode(timeline, "environment"),
    };
}

function nodeWidgetValues(node, names) {
    return Object.fromEntries(names.map((name) => [name, widgetValue(node, name, "")]));
}

function projectAssetPath(value) {
    const path = String(value || "").trim().replaceAll("\\", "/");
    if (!path || path.startsWith("/") || path.includes(":") || path.split("/").includes("..")) {
        throw new Error(`资源路径无效：${value || "空路径"}`);
    }
    return path;
}

const CRC32_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let value = 0; value < 256; value++) {
        let crc = value;
        for (let bit = 0; bit < 8; bit++) crc = (crc & 1) ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
        table[value] = crc >>> 0;
    }
    return table;
})();

function crc32(bytes) {
    let crc = 0xffffffff;
    for (const byte of bytes) crc = CRC32_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
}

function joinBytes(parts) {
    const result = new Uint8Array(parts.reduce((size, part) => size + part.length, 0));
    let offset = 0;
    for (const part of parts) {
        result.set(part, offset);
        offset += part.length;
    }
    return result;
}

function zipProjectEntries(entries) {
    const encoder = new TextEncoder();
    const localParts = [];
    const centralParts = [];
    let offset = 0;
    for (const entry of entries) {
        const name = encoder.encode(entry.name);
        const data = entry.data instanceof Uint8Array ? entry.data : new Uint8Array(entry.data);
        const crc = crc32(data);
        const local = new Uint8Array(30 + name.length);
        const localView = new DataView(local.buffer);
        localView.setUint32(0, 0x04034b50, true);
        localView.setUint16(4, 20, true);
        localView.setUint16(6, 0x0800, true);
        localView.setUint16(8, 0, true);
        localView.setUint32(14, crc, true);
        localView.setUint32(18, data.length, true);
        localView.setUint32(22, data.length, true);
        localView.setUint16(26, name.length, true);
        local.set(name, 30);
        localParts.push(local, data);
        const central = new Uint8Array(46 + name.length);
        const centralView = new DataView(central.buffer);
        centralView.setUint32(0, 0x02014b50, true);
        centralView.setUint16(4, 20, true);
        centralView.setUint16(6, 20, true);
        centralView.setUint16(8, 0x0800, true);
        centralView.setUint16(10, 0, true);
        centralView.setUint32(16, crc, true);
        centralView.setUint32(20, data.length, true);
        centralView.setUint32(24, data.length, true);
        centralView.setUint16(28, name.length, true);
        centralView.setUint32(42, offset, true);
        central.set(name, 46);
        centralParts.push(central);
        offset += local.length + data.length;
    }
    const central = joinBytes(centralParts);
    const end = new Uint8Array(22);
    const endView = new DataView(end.buffer);
    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(8, entries.length, true);
    endView.setUint16(10, entries.length, true);
    endView.setUint32(12, central.length, true);
    endView.setUint32(16, offset, true);
    return joinBytes([...localParts, central, end]);
}

function unzipProjectEntries(buffer) {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    let endOffset = -1;
    for (let offset = bytes.length - 22; offset >= Math.max(0, bytes.length - 65557); offset--) {
        if (view.getUint32(offset, true) === 0x06054b50) {
            endOffset = offset;
            break;
        }
    }
    if (endOffset < 0) throw new Error("项目包 ZIP 目录损坏");
    const count = view.getUint16(endOffset + 10, true);
    let offset = view.getUint32(endOffset + 16, true);
    const decoder = new TextDecoder();
    const entries = new Map();
    for (let index = 0; index < count; index++) {
        if (view.getUint32(offset, true) !== 0x02014b50) throw new Error("项目包目录项损坏");
        const method = view.getUint16(offset + 10, true);
        if (method !== 0) throw new Error("项目包使用了不支持的压缩方式，请使用本节点包重新打包");
        const expectedCrc = view.getUint32(offset + 16, true);
        const size = view.getUint32(offset + 24, true);
        const nameLength = view.getUint16(offset + 28, true);
        const extraLength = view.getUint16(offset + 30, true);
        const commentLength = view.getUint16(offset + 32, true);
        const localOffset = view.getUint32(offset + 42, true);
        const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength));
        if (view.getUint32(localOffset, true) !== 0x04034b50) throw new Error(`项目包文件 ${name} 的数据头损坏`);
        const localNameLength = view.getUint16(localOffset + 26, true);
        const localExtraLength = view.getUint16(localOffset + 28, true);
        const dataOffset = localOffset + 30 + localNameLength + localExtraLength;
        const data = bytes.slice(dataOffset, dataOffset + size);
        if (crc32(data) !== expectedCrc) throw new Error(`项目包文件 ${name} 校验失败`);
        entries.set(name, data);
        offset += 46 + nameLength + extraLength + commentLength;
    }
    return entries;
}

function timelineNodes() {
    return (app.graph?._nodes || []).filter((node) => nodeType(node) === TYPES.timeline);
}

function syncImageLoaderOptions() {
    for (const node of app.graph?._nodes || []) {
        if (nodeType(node) !== TYPES.loadImage) continue;
        const value = String(widgetValue(node, "image", ""));
        if (value) addWidgetOption(node, "image", value);
    }
}

function loadStyle() {
    if (document.getElementById("h3-timeline-editor-style")) return;
    const link = document.createElement("link");
    link.id = "h3-timeline-editor-style";
    link.rel = "stylesheet";
    link.href = new URL("./timeline_editor.css", import.meta.url).href;
    document.head.appendChild(link);
}

class H3TimelineEditor {
    constructor() {
        loadStyle();
        this.timelineId = null;
        this.selection = null;
        this.pixelsPerSecond = 115;
        this.snap = 0.05;
        this.playhead = 0;
        this.playing = false;
        this.scrub = null;
        this.drag = null;
        this.dockResize = null;
        this.library = null;
        this.referenceAssets = null;
        this.resourceFilter = "";
        this.resourceModal = null;
        this.referenceAssetModal = false;
        this.trackModal = null;
        this.confirmResolver = null;
        this.createdOffset = 0;
        this.autoSave = localStorage.getItem("minimax_h3_timeline_autosave") !== "false";
        this.dockLayout = this.loadDockLayout();
        this.saveTimer = null;
        this.changeVersion = 0;
        this.root = document.createElement("div");
        this.root.className = "h3te-root";
        this.root.tabIndex = -1;
        this.root.innerHTML = this.shell();
        document.body.appendChild(this.root);
        this.applyDockLayout();
        this.bind();
    }

    shell() {
        return `
            <header class="h3te-header">
                <div class="h3te-header-row h3te-header-primary">
                    <div class="h3te-brand"><span class="h3te-mark">H3</span><div><strong>导演时间轴</strong><small>MiniMax H3 Prompt Builder</small></div></div>
                    <label class="h3te-timeline-select">总时间轴 <select data-role="timeline"></select></label>
                    <div class="h3te-transport">
                        <button data-action="play" title="播放时间指针">▶</button>
                        <span data-role="time">00:00:00:00</span>
                    </div>
                    <button class="h3te-close" data-action="close">关闭</button>
                </div>
                <div class="h3te-header-row h3te-header-secondary">
                    <div class="h3te-tools">
                        <label>时长 <input class="h3te-duration" data-role="duration" type="number" min="0.21" max="60" step="0.05"></label>
                        <label>提示词 <select data-role="prompt-language"><option value="英文">英文</option><option value="中文">中文</option></select></label>
                        <label>吸附 <select data-role="snap"><option value="0">关闭</option><option value="0.01">0.01秒</option><option value="0.05" selected>0.05秒</option><option value="0.1">0.1秒</option><option value="0.25">0.25秒</option></select></label>
                        <label>缩放 <input data-role="zoom" type="range" min="60" max="260" value="115"></label>
                        <button data-action="fit">适应宽度</button><button data-action="bootstrap">补全工程</button><button data-action="export-project">导出 JSON</button><button data-action="package-project">打包项目</button><button data-action="import-project">导入并覆盖</button><button data-action="cleanup">清理未使用节点</button><button data-action="create-track">＋轨道</button>
                        <label class="h3te-autosave"><input data-role="autosave" type="checkbox" ${this.autoSave ? "checked" : ""}>自动保存</label><button data-action="save">保存</button><button data-action="refresh">刷新</button>
                        <input data-role="project-file" type="file" accept=".json,.h3proj,.h3proj.json,application/json,application/zip" hidden>
                    </div>
                </div>
            </header>
            <main class="h3te-main" data-role="dock-root">
                <aside class="h3te-dock-panel h3te-library" data-dock-panel="scene" data-dock-title="当前工程"><div data-role="library-scene"></div></aside>
                <aside class="h3te-dock-panel h3te-library" data-dock-panel="cards" data-dock-title="卡片"><div data-role="library-resources"></div></aside>
                <aside class="h3te-dock-panel h3te-library" data-dock-panel="media" data-dock-title="参考视频"><div data-role="library-media"></div></aside>
                <section class="h3te-dock-panel h3te-stage" data-dock-panel="timeline" data-dock-title="轨道时间轴">
                    <div class="h3te-ruler-wrap"><div class="h3te-track-label h3te-ruler-label">轨道</div><div class="h3te-ruler-scroll"><div class="h3te-ruler" data-role="ruler"></div></div></div>
                    <div class="h3te-tracks" data-role="tracks"></div>
                </section>
                <aside class="h3te-dock-panel h3te-inspector" data-dock-panel="inspector" data-dock-title="属性"><div data-role="inspector" class="h3te-inspector-body"></div></aside>
            </main>
            <footer class="h3te-footer"><span data-role="status">修改会直接同步到 ComfyUI 工作流</span><span data-role="save-status">${this.autoSave ? "自动保存已开启" : "自动保存已关闭"}</span><span>拖动播放头预览 · 空格播放 · ←/→逐帧</span></footer>
            <div class="h3te-modal-layer" data-role="resource-modal"></div>
            <div class="h3te-modal-layer" data-role="reference-asset-modal"></div>
            <div class="h3te-modal-layer" data-role="track-modal"></div>
            <div class="h3te-modal-layer" data-role="confirm-modal"></div>`;
    }

    defaultDockLayout() {
        return {
            version: 2,
            root: { type: "split", id: "split_root", direction: "row", sizes: [18, 57, 25], children: [
                { type: "group", id: "group_resources", panels: ["scene", "cards", "media"], active: "scene" },
                { type: "group", id: "group_timeline", panels: ["timeline"], active: "timeline" },
                { type: "group", id: "group_inspector", panels: ["inspector"], active: "inspector" },
            ] },
        };
    }

    loadDockLayout() {
        try {
            const stored = JSON.parse(localStorage.getItem("minimax_h3_timeline_docks") || "null");
            if (stored?.version === 2 && stored.root) return stored;
        } catch {
        }
        return this.defaultDockLayout();
    }

    saveDockLayout() {
        localStorage.setItem("minimax_h3_timeline_docks", JSON.stringify(this.dockLayout));
    }

    applyDockLayout() {
        const panels = new Map([...this.root.querySelectorAll("[data-dock-panel]")].map((panel) => [panel.dataset.dockPanel, panel]));
        const root = $("[data-role=dock-root]", this.root);
        root.innerHTML = "";
        const renderNode = (layout) => {
            if (layout.type === "group") {
                const group = document.createElement("section");
                group.className = "h3te-dock-group";
                group.dataset.dockGroup = layout.id;
                if (!layout.panels.includes(layout.active)) layout.active = layout.panels[0] || null;
                group.innerHTML = `<div class="h3te-dock-tabs">${layout.panels.map((id) => {
                    const panel = panels.get(id);
                    return `<button class="h3te-dock-tab ${id === layout.active ? "is-active" : ""}" draggable="true" data-dock-tab="${id}">${escapeHtml(panel?.dataset.dockTitle || id)}</button>`;
                }).join("")}</div><div class="h3te-dock-body"></div><div class="h3te-dock-guide"><i data-dock-position="top"></i><i data-dock-position="left"></i><i data-dock-position="center"></i><i data-dock-position="right"></i><i data-dock-position="bottom"></i></div>`;
                const body = $(".h3te-dock-body", group);
                for (const id of layout.panels) {
                    const panel = panels.get(id);
                    if (!panel) continue;
                    body.appendChild(panel);
                    panel.hidden = id !== layout.active;
                }
                return group;
            }
            const split = document.createElement("div");
            split.className = `h3te-dock-split ${layout.direction}`;
            split.dataset.dockSplit = layout.id;
            const total = layout.sizes.reduce((sum, value) => sum + value, 0) || layout.children.length;
            layout.children.forEach((child, index) => {
                const cell = document.createElement("div");
                cell.className = "h3te-dock-cell";
                cell.style.flexBasis = `${100 * (layout.sizes[index] || 1) / total}%`;
                cell.appendChild(renderNode(child));
                split.appendChild(cell);
                if (index < layout.children.length - 1) {
                    const resizer = document.createElement("div");
                    resizer.className = "h3te-dock-resizer";
                    resizer.dataset.dockResize = `${layout.id}:${index}`;
                    split.appendChild(resizer);
                }
            });
            return split;
        };
        root.appendChild(renderNode(this.dockLayout.root));
        this.saveDockLayout();
    }

    findDockNode(id, node = this.dockLayout.root) {
        if (node.id === id) return node;
        if (node.type === "split") for (const child of node.children) {
            const found = this.findDockNode(id, child);
            if (found) return found;
        }
        return null;
    }

    findDockParent(id, node = this.dockLayout.root) {
        if (node.type !== "split") return null;
        const index = node.children.findIndex((child) => child.id === id);
        if (index >= 0) return { node, index };
        for (const child of node.children) {
            const found = this.findDockParent(id, child);
            if (found) return found;
        }
        return null;
    }

    removeDockPanel(panelId, node = this.dockLayout.root) {
        if (node.type === "group") {
            node.panels = node.panels.filter((id) => id !== panelId);
            if (node.active === panelId) node.active = node.panels[0] || null;
            return;
        }
        for (const child of node.children) this.removeDockPanel(panelId, child);
    }

    normalizeDockNode(node) {
        if (node.type === "group") return node.panels.length ? node : null;
        const entries = node.children.map((child, index) => ({ child: this.normalizeDockNode(child), size: node.sizes[index] }))
            .filter((entry) => entry.child);
        if (!entries.length) return null;
        if (entries.length === 1) return entries[0].child;
        node.children = entries.map((entry) => entry.child);
        node.sizes = entries.map((entry) => entry.size || 1);
        return node;
    }

    dockPanel(panelId, groupId, position = "center", beforeId = null) {
        const valid = ["scene", "cards", "media", "timeline", "inspector"];
        const target = this.findDockNode(groupId);
        if (!valid.includes(panelId) || target?.type !== "group") return;
        this.removeDockPanel(panelId);
        if (position === "center") {
            const index = beforeId ? target.panels.indexOf(beforeId) : -1;
            if (index >= 0) target.panels.splice(index, 0, panelId);
            else target.panels.push(panelId);
            target.active = panelId;
        } else {
            const direction = ["left", "right"].includes(position) ? "row" : "column";
            const group = { type: "group", id: `group_${crypto.randomUUID()}`, panels: [panelId], active: panelId };
            const first = ["left", "top"].includes(position) ? group : target;
            const second = first === group ? target : group;
            const split = { type: "split", id: `split_${crypto.randomUUID()}`, direction, sizes: [1, 1], children: [first, second] };
            const parent = this.findDockParent(target.id);
            if (parent) parent.node.children[parent.index] = split;
            else this.dockLayout.root = split;
        }
        this.dockLayout.root = this.normalizeDockNode(this.dockLayout.root);
        this.applyDockLayout();
        this.renderTracks();
    }

    activateDockPanel(panelId) {
        const find = (node) => node.type === "group" ? (node.panels.includes(panelId) ? node : null)
            : node.children.map(find).find(Boolean);
        const group = find(this.dockLayout.root);
        if (!group) return;
        group.active = panelId;
        this.applyDockLayout();
        if (panelId === "timeline") this.renderTracks();
    }

    bind() {
        this.root.addEventListener("click", (event) => this.onClick(event));
        this.root.addEventListener("dragstart", (event) => {
            const tab = event.target.closest("[data-dock-tab]");
            if (!tab) return;
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/minimax-h3-panel", tab.dataset.dockTab);
            tab.classList.add("is-dragging");
        });
        this.root.addEventListener("dragover", (event) => {
            const group = event.target.closest("[data-dock-group]");
            if (!group || !event.dataTransfer.types.includes("text/minimax-h3-panel")) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
            for (const item of this.root.querySelectorAll(".h3te-dock-group.is-drop-target")) item.classList.remove("is-drop-target");
            const rect = group.getBoundingClientRect();
            const x = (event.clientX - rect.left) / Math.max(1, rect.width);
            const y = (event.clientY - rect.top) / Math.max(1, rect.height);
            let position = "center";
            if (!event.target.closest(".h3te-dock-tabs")) {
                if (x < 0.25) position = "left";
                else if (x > 0.75) position = "right";
                else if (y < 0.25) position = "top";
                else if (y > 0.75) position = "bottom";
            }
            group.dataset.dropPosition = position;
            group.classList.add("is-drop-target");
        });
        this.root.addEventListener("drop", (event) => {
            const group = event.target.closest("[data-dock-group]");
            const panelId = event.dataTransfer.getData("text/minimax-h3-panel");
            if (!group || !panelId) return;
            event.preventDefault();
            const before = event.target.closest("[data-dock-tab]")?.dataset.dockTab || null;
            this.dockPanel(panelId, group.dataset.dockGroup, group.dataset.dropPosition || "center", before === panelId ? null : before);
            group.classList.remove("is-drop-target");
        });
        this.root.addEventListener("dragend", () => {
            for (const item of this.root.querySelectorAll(".is-dragging,.is-drop-target")) item.classList.remove("is-dragging", "is-drop-target");
        });
        $("[data-role=timeline]", this.root).addEventListener("change", (event) => {
            this.timelineId = Number(event.target.value);
            this.selection = null;
            this.layoutProjectNodes(true);
            this.render();
        });
        $("[data-role=zoom]", this.root).addEventListener("input", (event) => {
            this.pixelsPerSecond = Number(event.target.value);
            this.renderTracks();
        });
        $("[data-role=snap]", this.root).addEventListener("change", (event) => this.snap = Number(event.target.value));
        $("[data-role=prompt-language]", this.root).addEventListener("change", (event) => {
            const timeline = this.currentTimeline();
            if (!timeline) return;
            app.graph.beforeChange?.();
            setWidgetValue(timeline, "prompt_language", event.target.value);
            app.graph.afterChange?.();
            this.markWorkflowChanged();
            this.setStatus(`提示词语言已切换为${event.target.value}`);
        });
        $("[data-role=project-file]", this.root).addEventListener("change", (event) => this.importProjectFile(event));
        $("[data-role=autosave]", this.root).addEventListener("change", (event) => {
            this.autoSave = event.target.checked;
            localStorage.setItem("minimax_h3_timeline_autosave", String(this.autoSave));
            if (this.autoSave) {
                this.setSaveStatus("自动保存已开启");
                if (this.changeVersion) this.scheduleAutoSave();
            } else {
                clearTimeout(this.saveTimer);
                this.saveTimer = null;
                this.setSaveStatus("自动保存已关闭");
            }
        });
        $("[data-role=duration]", this.root).addEventListener("change", (event) => {
            const timeline = this.currentTimeline();
            const value = Math.max(0.21, Math.min(60, Number(event.target.value) || 0.21));
            if (timeline && setWidgetValue(timeline, "duration_seconds", value)) {
                this.data = collectTimeline(timeline);
                this.playhead = Math.min(this.playhead, value);
                this.renderTracks();
                this.setStatus("总时长已同步到工作流");
            }
        });
        this.root.addEventListener("change", (event) => this.onFieldChange(event));
        this.root.addEventListener("input", (event) => {
            if (event.target.matches("[data-role=resource-search]")) {
                this.resourceFilter = event.target.value;
                this.renderLibrary();
                requestAnimationFrame(() => {
                    const search = $("[data-role=resource-search]", this.root);
                    search?.focus();
                    search?.setSelectionRange(this.resourceFilter.length, this.resourceFilter.length);
                });
            }
        });
        this.root.addEventListener("pointerdown", (event) => this.onPointerDown(event));
        this.root.addEventListener("loadedmetadata", (event) => {
            if (event.target.matches?.("[data-timeline-preview]")) this.syncTimelinePreview();
        }, true);
        window.addEventListener("pointermove", (event) => this.onPointerMove(event));
        window.addEventListener("pointerup", (event) => this.onPointerUp(event));
        window.addEventListener("keydown", (event) => {
            if (!this.root.classList.contains("is-open")) return;
            const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
            if (!typing && [" ", "ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                event.preventDefault();
                if (event.key === " ") return this.togglePlay();
                const frame = event.shiftKey ? 1 : 1 / 24;
                if (event.key === "ArrowLeft") this.setPlayhead(this.playhead - frame, { scrollIntoView: true });
                else if (event.key === "ArrowRight") this.setPlayhead(this.playhead + frame, { scrollIntoView: true });
                else if (event.key === "Home") this.setPlayhead(0, { scrollIntoView: true });
                else this.setPlayhead(this.data?.duration || 0, { scrollIntoView: true });
                return;
            }
            if (event.key !== "Escape") return;
            if (this.confirmResolver) {
                this.closeConfirm(false);
            } else if (this.resourceModal) {
                this.resourceModal = null;
                this.renderResourceModal();
            } else if (this.referenceAssetModal) {
                this.referenceAssetModal = false;
                this.renderReferenceAssetModal();
            } else if (this.trackModal) {
                this.trackModal = null;
                this.renderTrackModal();
            } else {
                this.close();
            }
        });
    }

    open(timeline = null) {
        const timelines = timelineNodes();
        if (timeline) this.timelineId = timeline.id;
        if (!timelines.some((node) => node.id === this.timelineId)) this.timelineId = timelines[0]?.id ?? null;
        syncImageLoaderOptions();
        this.migrateActionContextWidgets();
        this.migrateActorMacros();
        this.layoutProjectNodes(true);
        this.root.classList.add("is-open");
        this.root.focus({ preventScroll: true });
        document.body.classList.add("h3te-open");
        this.render();
        this.loadResourceLibrary();
        this.loadReferenceAssets();
    }

    migrateActionContextWidgets() {
        const nodes = (app.graph?._nodes || []).filter((node) => nodeType(node) === TYPES.action &&
            (node._minimaxH3ContextMigrated || typeof widget(node, "use_previous_context")?.value !== "boolean"));
        if (!nodes.length) return;
        app.graph.beforeChange?.();
        let changed = false;
        for (const node of nodes) {
            changed = migrateActionContextWidget(node) || node._minimaxH3ContextMigrated || changed;
            delete node._minimaxH3ContextMigrated;
        }
        app.graph.afterChange?.();
        if (!changed) return;
        app.graph.setDirtyCanvas(true, true);
        this.markWorkflowChanged();
        this.setStatus(`已修复 ${nodes.length} 个旧动作片段的段间引导字段`);
    }

    migrateActorMacros() {
        const timeline = this.currentTimeline();
        const group = linkedNode(timeline, "character_group");
        const actors = group ? numberedInputs(group, ["actors", "actor"]) : [];
        if (!actors.length) return;
        const used = new Set();
        const replacements = [];
        app.graph.beforeChange?.();
        let changed = false;
        for (const [index, actor] of actors.entries()) {
            let actorId = String(widgetValue(actor, "actor_id", "")).trim();
            if (!/^actor_[1-9][0-9]*$/.test(actorId) || used.has(actorId)) {
                let number = index + 1;
                while (used.has(`actor_${number}`)) number += 1;
                actorId = `actor_${number}`;
                changed = setWidgetValue(actor, "actor_id", actorId) || changed;
            }
            used.add(actorId);
            const name = String(widgetValue(findCardForActor(actor), "name", "")).trim();
            if (name) replacements.push([name, `{${actorId}}`]);
        }
        for (const node of upstreamGraphNodes(timeline)) {
            if (nodeType(node) === TYPES.character) continue;
            for (const [fieldName] of FIELD_SETS[nodeType(node)] || []) {
                if (fieldName === "actor_id") continue;
                const item = widget(node, fieldName);
                if (!item || typeof item.value !== "string") continue;
                let value = item.value;
                for (const [name, token] of replacements) {
                    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                    value = value.replace(new RegExp(`(^|[^\\p{L}\\p{N}_])${escaped}(?=$|[^\\p{L}\\p{N}_])`, "giu"), `$1${token}`);
                }
                if (value !== item.value) changed = setWidgetValue(node, fieldName, value) || changed;
            }
        }
        app.graph.afterChange?.();
        if (!changed) return;
        app.graph.setDirtyCanvas(true, true);
        this.markWorkflowChanged();
        this.setStatus("已将旧人物名称引用迁移为人物实例宏");
    }

    close() {
        this.playing = false;
        this.closeConfirm(false);
        this.resourceModal = null;
        this.referenceAssetModal = false;
        this.trackModal = null;
        this.renderResourceModal();
        this.renderReferenceAssetModal();
        this.renderTrackModal();
        this.root.classList.remove("is-open");
        document.body.classList.remove("h3te-open");
    }

    confirmDelete(message, title = "确认删除", confirmText = "删除") {
        this.closeConfirm(false);
        const host = $("[data-role=confirm-modal]", this.root);
        host.classList.add("is-open");
        host.innerHTML = `<div class="h3te-confirm-box"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p><div><button data-action="cancel-delete">取消</button><button class="danger" data-action="confirm-delete">${escapeHtml(confirmText)}</button></div></div>`;
        return new Promise((resolve) => this.confirmResolver = resolve);
    }

    closeConfirm(result) {
        if (!this.confirmResolver) return;
        const resolve = this.confirmResolver;
        this.confirmResolver = null;
        const host = $("[data-role=confirm-modal]", this.root);
        host.classList.remove("is-open");
        host.innerHTML = "";
        resolve(result);
    }

    currentTimeline() {
        return app.graph?.getNodeById(this.timelineId);
    }

    nextActorMacro(timeline = this.currentTimeline()) {
        const group = linkedNode(timeline, "character_group");
        const used = new Set((group ? numberedInputs(group, ["actors", "actor"]) : [])
            .map((actor) => String(widgetValue(actor, "actor_id", ""))));
        let number = 1;
        while (used.has(`actor_${number}`)) number += 1;
        return `actor_${number}`;
    }

    serializeProject() {
        const timeline = this.currentTimeline();
        if (!timeline) throw new Error("当前没有可导出的总时间轴");
        timeline.properties ||= {};
        timeline.properties.minimax_h3_project ||= { id: crypto.randomUUID(), extensions: {} };
        const data = collectTimeline(timeline);
        const assets = [];
        const assetIds = new Map();
        const addAsset = (kind, path) => {
            if (!path) return null;
            path = projectAssetPath(path);
            const key = `${kind}:${path}`;
            if (assetIds.has(key)) return assetIds.get(key);
            const id = `${kind}_${assets.length + 1}`;
            assetIds.set(key, id);
            assets.push({ id, kind, path });
            return id;
        };
        const imageAsset = (card) => {
            const loader = linkedNode(card, "reference_image");
            return nodeType(loader) === TYPES.loadImage ? addAsset("image", widgetValue(loader, "image", "")) : null;
        };
        const videoAsset = (consumer, inputName) => {
            const source = linkedNode(consumer, inputName);
            return nodeType(source) === TYPES.loadVideo ? addAsset("video", widgetValue(source, "file", "")) : null;
        };
        const actors = data.characterGroup ? numberedInputs(data.characterGroup, ["actors", "actor"]) : [];
        const actorIds = new Map(actors.map((actor, index) => [actor.id, `actor_${index + 1}`]));
        const characters = actors.map((actor, index) => {
            const card = findCardForActor(actor);
            return {
                id: `actor_${index + 1}`,
                card: nodeWidgetValues(card, ["name", "description", "style_priority", "character_style"]),
                instance: nodeWidgetValues(actor, ["actor_id", "position_override", "pose_override", "emotion_override", "appearance_override"]),
                reference_asset: imageAsset(card),
            };
        });
        const style = data.style ? {
            card: nodeWidgetValues(data.style, ["style", "rendering", "color_palette", "texture", "reference_usage"]),
            reference_asset: imageAsset(data.style),
        } : null;
        const environmentCard = findCardForEnvironment(data.environment);
        const environment = data.environment ? {
            card: nodeWidgetValues(environmentCard, ["name", "location", "default_background"]),
            instance: nodeWidgetValues(data.environment, ["location_override", "time_weather_override", "background_override", "atmosphere_override"]),
            reference_asset: imageAsset(environmentCard),
        } : null;
        const tracks = data.tracks.map((track, trackIndex) => ({
            id: `track_${trackIndex + 1}`,
            type: track.type === TYPES.actorTrack ? "actor" : track.type === TYPES.environmentTrack ? "environment" : "system",
            owner: track.owner ? actorIds.get(track.owner.id) || "environment" : null,
            clips: track.clips.map((clip, clipIndex) => {
                const type = nodeType(clip.node);
                const fields = nodeWidgetValues(clip.node, (FIELD_SETS[type] || []).map(([name]) => name));
                const target = linkedNode(clip.node, "target");
                const language = linkedNode(clip.node, "language");
                const config = REFERENCE_SLOTS[type];
                const reference = config ? linkedNode(clip.node, config.input) : null;
                const split = nodeType(reference) === TYPES.videoPerson ? linkedNode(reference, "motion_reference") : reference;
                const referenceAsset = split ? videoAsset(split, "video") : null;
                const resultAsset = clip.resultNode ? videoAsset(clip.resultNode, "video") : null;
                return {
                    id: `track_${trackIndex + 1}_clip_${clipIndex + 1}`,
                    type: Object.entries(PROJECT_CLIP_TYPES).find(([, nodeTypeId]) => nodeTypeId === type)?.[0],
                    fields,
                    target: target ? actorIds.get(target.id) || null : null,
                    language: language ? nodeWidgetValues(language, ["language", "variant", "accent", "pronunciation"]) : null,
                    reference: referenceAsset ? {
                        asset: referenceAsset,
                        trim_start: Number(widgetValue(split, "trim_start", 0)) || 0,
                        trim_end: Number(widgetValue(split, "trim_end", 0)) || 0,
                        ...(nodeType(reference) === TYPES.videoPerson ? {
                            person_id: widgetValue(reference, "person_id", ""),
                            person_description: widgetValue(reference, "person_description", ""),
                        } : {}),
                    } : null,
                    cached_result: resultAsset ? {
                        asset: resultAsset,
                        version: Number(widgetValue(clip.resultNode, "result_version", 0)) || 0,
                    } : null,
                };
            }),
        }));
        return {
            format: PROJECT_FORMAT,
            version: PROJECT_VERSION,
            project: {
                id: timeline.properties.minimax_h3_project.id,
                name: timeline.title || "MiniMax H3 项目",
                duration_seconds: data.duration,
                prompt_language: data.promptLanguage,
            },
            assets,
            characters,
            style,
            environment,
            tracks,
            extensions: structuredClone(timeline.properties.minimax_h3_project.extensions || {}),
        };
    }

    exportProject() {
        try {
            const project = this.serializeProject();
            const blob = new Blob([`${JSON.stringify(project, null, 2)}\n`], { type: "application/json" });
            this.downloadProjectBlob(blob, project.project.name, ".h3proj.json");
        } catch (error) {
            this.setStatus(`项目导出失败：${error?.message || error}`);
        }
    }

    downloadProjectBlob(blob, projectName, extension) {
        const link = document.createElement("a");
        const name = String(projectName || "minimax-h3-project").replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").trim() || "minimax-h3-project";
        link.href = URL.createObjectURL(blob);
        link.download = `${name}${extension}`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
        this.setStatus(`项目已导出：${link.download}`);
    }

    async packageProject() {
        try {
            const project = this.serializeProject();
            const packaged = structuredClone(project);
            const entries = [];
            for (const asset of packaged.assets) {
                this.setStatus(`正在打包资源：${asset.path}`);
                const response = await fetch(this.referenceVideoUrl(asset.path));
                if (!response.ok) throw new Error(`读取资源失败（HTTP ${response.status}）：${asset.path}`);
                const extension = asset.path.match(/(\.[a-z0-9]{1,10})$/i)?.[1]?.toLowerCase() || (asset.kind === "image" ? ".png" : ".mp4");
                const safeId = String(asset.id).replace(/[^a-zA-Z0-9_-]/g, "_");
                asset.source_path = asset.path;
                asset.path = `assets/${safeId}${extension}`;
                entries.push({ name: asset.path, data: new Uint8Array(await response.arrayBuffer()) });
            }
            entries.unshift({ name: "project.json", data: new TextEncoder().encode(`${JSON.stringify(packaged, null, 2)}\n`) });
            const archive = zipProjectEntries(entries);
            this.downloadProjectBlob(new Blob([archive], { type: "application/zip" }), project.project.name, ".h3proj");
        } catch (error) {
            this.setStatus(`项目打包失败：${error?.message || error}`);
        }
    }

    validateProject(rawProject) {
        const project = structuredClone(rawProject);
        if (!project || project.format !== PROJECT_FORMAT) throw new Error("不是 MiniMax H3 导演项目文件");
        if (project.version !== PROJECT_VERSION) throw new Error(`不支持项目格式版本 ${project.version}`);
        const duration = Number(project.project?.duration_seconds);
        if (!(duration >= 0.21 && duration <= 60)) throw new Error("项目时长必须在 0.21 到 60 秒之间");
        if (!Array.isArray(project.characters) || !project.characters.length) throw new Error("项目至少需要一个人物");
        if (!project.style || !project.environment || !Array.isArray(project.tracks)) throw new Error("项目缺少风格、环境或轨道数据");
        const assets = new Map();
        for (const asset of project.assets || []) {
            if (!asset?.id || !["image", "video"].includes(asset.kind) || assets.has(asset.id)) throw new Error("资源清单包含无效或重复 ID");
            assets.set(asset.id, { ...asset, path: projectAssetPath(asset.path) });
        }
        const actorIds = new Set();
        const actorMacros = new Set();
        const nameReplacements = [];
        for (const [index, actor] of project.characters.entries()) {
            if (!actor?.id || actorIds.has(actor.id)) throw new Error("人物包含无效或重复 ID");
            actorIds.add(actor.id);
            actor.instance ||= {};
            let actorId = String(actor.instance.actor_id || "").trim();
            if (!/^actor_[1-9][0-9]*$/.test(actorId) || actorMacros.has(actorId)) {
                let number = index + 1;
                while (actorMacros.has(`actor_${number}`)) number += 1;
                actorId = `actor_${number}`;
            }
            actor.instance.actor_id = actorId;
            actorMacros.add(actorId);
            const name = String(actor.card?.name || "").trim();
            if (name) nameReplacements.push([name, `{${actorId}}`]);
        }
        const migrateText = (value) => {
            if (typeof value !== "string") return value;
            for (const [name, token] of nameReplacements) {
                const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                value = value.replace(new RegExp(`(^|[^\\p{L}\\p{N}_])${escaped}(?=$|[^\\p{L}\\p{N}_])`, "giu"), `$1${token}`);
            }
            return value;
        };
        const migrateObject = (object) => {
            for (const [key, value] of Object.entries(object || {})) {
                if (key === "actor_id") continue;
                if (typeof value === "string") object[key] = migrateText(value);
            }
        };
        for (const actor of project.characters) migrateObject(actor.instance);
        migrateObject(project.style.card);
        migrateObject(project.environment.card);
        migrateObject(project.environment.instance);
        const allowedTypes = new Set(Object.keys(PROJECT_CLIP_TYPES));
        const requireAsset = (id, kind, label) => {
            if (!id) return;
            const asset = assets.get(id);
            if (!asset) throw new Error(`${label}引用了不存在的资源 ${id}`);
            if (asset.kind !== kind) throw new Error(`${label}必须引用${kind === "image" ? "图片" : "视频"}资源`);
        };
        for (const actor of project.characters) requireAsset(actor.reference_asset, "image", `人物 ${actor.id}`);
        requireAsset(project.style.reference_asset, "image", "风格卡");
        requireAsset(project.environment.reference_asset, "image", "环境卡");
        for (const track of project.tracks) {
            if (!["actor", "environment", "system"].includes(track?.type) || !Array.isArray(track.clips)) throw new Error("轨道结构无效");
            if (track.type === "actor" && !actorIds.has(track.owner)) throw new Error(`人物轨道引用了不存在的人物 ${track.owner}`);
            const clipKinds = new Set();
            for (const clip of track.clips) {
                const legacyType = Object.entries(PROJECT_CLIP_TYPES).find(([, nodeTypeId]) => nodeTypeId === clip?.type)?.[0];
                if (legacyType) clip.type = legacyType;
                if (!allowedTypes.has(clip?.type)) throw new Error(`不支持片段类型 ${clip?.type}`);
                migrateObject(clip.fields);
                for (const value of Object.values(clip.fields || {})) {
                    if (typeof value !== "string") continue;
                    for (const match of value.matchAll(/\{(actor[^}]*)\}/g)) {
                        if (!actorMacros.has(match[1])) throw new Error(`片段 ${clip.id || "未命名"} 使用了未声明的人物宏 ${match[0]}`);
                    }
                }
                clipKinds.add(clip.type);
                if (track.type === "actor" && clip.type !== "action") throw new Error("人物轨道只能包含人物动作片段");
                if (track.type === "environment" && clip.type !== "environment") throw new Error("环境轨道只能包含环境片段");
                if (track.type === "system" && !["camera", "lighting", "audio"].includes(clip.type)) throw new Error("系统轨道只能包含镜头、灯光或音频片段");
                if (clip.type === "action") {
                    clip.fields.action_type ||= "body";
                    if (!["body", "expression", "gaze", "speech"].includes(clip.fields.action_type)) {
                        throw new Error(`人物动作片段 ${clip.id || "未命名"} 的动作种类无效`);
                    }
                }
                const start = Number(clip.fields?.start_time);
                const end = Number(clip.fields?.end_time);
                if (!(start >= 0 && end > start && end <= duration + 1e-6)) throw new Error(`片段 ${clip.id || "未命名"} 的时间范围无效`);
                if (clip.target && !actorIds.has(clip.target)) throw new Error(`片段目标人物不存在：${clip.target}`);
                if (clip.type === "action" && clip.fields?.action_type === "speech" && !clip.language) throw new Error(`对话片段 ${clip.id || "未命名"} 缺少语言数据`);
                requireAsset(clip.reference?.asset, "video", `片段 ${clip.id || "未命名"} 的参考视频`);
                if (clip.reference && (clip.reference.person_id || clip.reference.person_description)) {
                    if (clip.type !== "action" || !/^person_[1-9][0-9]*$/.test(clip.reference.person_id || "") || !String(clip.reference.person_description || "").trim()) {
                        throw new Error(`片段 ${clip.id || "未命名"} 的参考视频人物解释无效`);
                    }
                }
                requireAsset(clip.cached_result?.asset, "video", `片段 ${clip.id || "未命名"} 的缓存结果`);
            }
            if (track.type === "system" && clipKinds.size > 1) throw new Error("同一系统轨道不能混合镜头、灯光和音频片段");
        }
        return { project, assets };
    }

    async importProjectFile(event) {
        const input = event.currentTarget;
        const file = input.files?.[0];
        input.value = "";
        if (!file) return;
        try {
            const timeline = this.currentTimeline();
            if (!timeline) throw new Error("请先创建或选择一个要覆盖的总时间轴");
            if (!await this.confirmDelete(`用“${file.name}”覆盖当前项目“${timeline.title || `总时间轴 #${timeline.id}`}”？总时间轴节点及其全部下游生成连线会保留，现有卡片、轨道和片段会被替换。`, "确认覆盖", "覆盖")) return;
            const bytes = new Uint8Array(await file.arrayBuffer());
            const isZip = bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b && bytes[2] === 0x03 && bytes[3] === 0x04;
            const project = isZip ? await this.unpackProject(bytes.buffer) : JSON.parse(new TextDecoder().decode(bytes));
            this.importProject(project);
        } catch (error) {
            this.setStatus(`项目导入失败：${error?.message || error}`);
        }
    }

    async unpackProject(buffer) {
        const entries = unzipProjectEntries(buffer);
        const manifest = entries.get("project.json");
        if (!manifest) throw new Error("项目包缺少 project.json");
        const project = JSON.parse(new TextDecoder().decode(manifest));
        if (!Array.isArray(project.assets)) throw new Error("项目包资源清单无效");
        const projectId = String(project.project?.id || "imported").replace(/[^a-zA-Z0-9_-]/g, "_");
        for (const asset of project.assets) {
            const data = entries.get(projectAssetPath(asset.path));
            if (!data) throw new Error(`项目包缺少资源文件：${asset.path}`);
            const originalName = asset.path.split("/").pop();
            const file = new File([data], originalName, { type: asset.kind === "image" ? "image/*" : "video/mp4" });
            const body = new FormData();
            body.append("image", file, originalName);
            body.append("type", "input");
            body.append("subfolder", `minimax_h3/projects/${projectId}`);
            const response = await api.fetchApi("/upload/image", { method: "POST", body });
            if (!response.ok) throw new Error(`恢复资源失败（HTTP ${response.status}）：${originalName}`);
            const uploaded = await response.json();
            asset.path = uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name || uploaded.filename}` : uploaded.name || uploaded.filename;
        }
        return project;
    }

    importProject(rawProject) {
        const { project, assets } = this.validateProject(rawProject);
        const timeline = this.currentTimeline();
        if (!timeline) throw new Error("请先创建或选择一个要覆盖的总时间轴");
        const created = [];
        const anchor = timeline.pos;
        const inputNames = ["character_group", "style_card", "environment", "tracks"];
        const oldRoots = new Map(inputNames.map((name) => [name, linkedNode(timeline, name)]));
        const oldUpstream = upstreamGraphNodes(timeline);
        const oldState = {
            title: timeline.title,
            properties: structuredClone(timeline.properties || {}),
            duration: widgetValue(timeline, "duration_seconds", 5),
            language: widgetValue(timeline, "prompt_language", "英文"),
        };
        let swapped = false;
        this.pendingCreatedNodes = created;
        app.graph.beforeChange?.();
        try {
            const mediaNodes = new Map();
            const mediaNode = (assetId) => {
                if (!assetId) return null;
                if (mediaNodes.has(assetId)) return mediaNodes.get(assetId);
                const asset = assets.get(assetId);
                const loader = this.createNode(asset.kind === "image" ? TYPES.loadImage : TYPES.loadVideo, [anchor[0] - 2000, anchor[1]],
                    asset.kind === "image" ? {} : { file: asset.path });
                loader.properties ||= {};
                loader.properties.minimax_h3_managed_media = true;
                if (asset.kind === "image") setImageLoaderValue(loader, asset.path);
                mediaNodes.set(assetId, loader);
                return loader;
            };
            const referenceNodes = new Map();
            const referenceNode = (binding) => {
                const start = Number(binding.trim_start) || 0;
                const end = Number(binding.trim_end) || 0;
                const key = `${binding.asset}\u0000${start}\u0000${end}`;
                if (referenceNodes.has(key)) return { key, node: referenceNodes.get(key) };
                const loader = mediaNode(binding.asset);
                const reference = this.createNode(TYPES.motionReference, [anchor[0] - 1400, anchor[1]], {
                    trim_start: start,
                    trim_end: end,
                });
                if (!connectNodes(loader, loader.outputs?.[0]?.name, reference, "video")) throw new Error("无法连接片段参考视频");
                referenceNodes.set(key, reference);
                return { key, node: reference };
            };
            const personReferenceNodes = new Map();
            const group = this.createNode(TYPES.characterGroup, [anchor[0] - 700, anchor[1]]);
            const actorNodes = new Map();
            for (const [actorIndex, actorData] of project.characters.entries()) {
                const card = this.createNode(TYPES.character, [anchor[0] - 1600, anchor[1]], actorData.card || {});
                const actor = this.createNode(TYPES.actor, [anchor[0] - 1200, anchor[1]], {
                    ...(actorData.instance || {}),
                    actor_id: actorData.instance?.actor_id || (/^actor_[1-9][0-9]*$/.test(actorData.id) ? actorData.id : `actor_${actorIndex + 1}`),
                });
                const image = mediaNode(actorData.reference_asset);
                if (image && !connectNodes(image, image.outputs?.[0]?.name, card, "reference_image")) throw new Error("无法连接人物参考图");
                if (!connectNodes(card, "character_card", actor, "character_card") || !this.connectAutogrow(actor, "actor_instance", group, "actor")) throw new Error("无法连接人物数据");
                actorNodes.set(actorData.id, actor);
            }
            const style = this.createNode(TYPES.visual, [anchor[0] - 1600, anchor[1]], project.style.card || {});
            const styleImage = mediaNode(project.style.reference_asset);
            if (styleImage && !connectNodes(styleImage, styleImage.outputs?.[0]?.name, style, "reference_image")) throw new Error("无法连接风格参考图");
            const environmentCard = this.createNode(TYPES.environment, [anchor[0] - 1600, anchor[1]], project.environment.card || {});
            const environment = this.createNode(TYPES.environmentInstance, [anchor[0] - 1200, anchor[1]], project.environment.instance || {});
            const environmentImage = mediaNode(project.environment.reference_asset);
            if (environmentImage && !connectNodes(environmentImage, environmentImage.outputs?.[0]?.name, environmentCard, "reference_image")) throw new Error("无法连接环境参考图");
            if (!connectNodes(environmentCard, "environment_card", environment, "environment_card")) throw new Error("无法连接环境数据");
            const trackList = this.createNode(TYPES.trackList, [anchor[0] - 400, anchor[1]]);
            for (const trackData of project.tracks) {
                const trackType = trackData.type === "actor" ? TYPES.actorTrack : trackData.type === "environment" ? TYPES.environmentTrack : TYPES.systemTrack;
                const track = this.createNode(trackType, [anchor[0] - 800, anchor[1]]);
                if (trackData.type === "actor" && !connectNodes(actorNodes.get(trackData.owner), "actor_instance", track, "actor")) throw new Error("无法连接人物轨道");
                if (trackData.type === "environment" && !connectNodes(environment, "environment_instance", track, "environment")) throw new Error("无法连接环境轨道");
                for (const clipData of trackData.clips) {
                    const clipType = PROJECT_CLIP_TYPES[clipData.type];
                    const clip = this.createNode(clipType, [anchor[0] - 1100, anchor[1]], clipData.fields || {});
                    if (clipData.type === "action") {
                        const actionType = clipData.fields?.action_type || "body";
                        if (widgetValue(clip, "action_type", "") !== actionType) setWidgetValue(clip, "action_type", actionType);
                        if (widgetValue(clip, "action_type", "") !== actionType) throw new Error(`无法设置人物动作种类 ${actionType}`);
                    }
                    if (clipData.target && !connectNodes(actorNodes.get(clipData.target), "actor_instance", clip, "target")) throw new Error("无法连接片段目标人物");
                    if (clipData.language) {
                        const language = this.createNode(TYPES.language, [anchor[0] - 1400, anchor[1]], clipData.language);
                        if (!connectNodes(language, "language", clip, "language")) throw new Error("无法连接对话语言");
                    }
                    const config = REFERENCE_SLOTS[clipType];
                    if (clipData.reference && config) {
                        const split = referenceNode(clipData.reference);
                        const reference = split.node;
                        let semanticNode = reference;
                        let semanticOutput = config.output;
                        if (clipData.type === "action" && clipData.reference.person_description) {
                            const personKey = `${split.key}\u0000${clipData.reference.person_id}\u0000${clipData.reference.person_description}`;
                            semanticNode = personReferenceNodes.get(personKey);
                            if (!semanticNode) {
                                semanticNode = this.createNode(TYPES.videoPerson, [anchor[0] - 1200, anchor[1]], {
                                    person_id: clipData.reference.person_id,
                                    person_description: clipData.reference.person_description,
                                });
                                if (!connectNodes(reference, reference.outputs?.[0]?.name, semanticNode, "motion_reference")) throw new Error("无法连接参考视频人物解释");
                                personReferenceNodes.set(personKey, semanticNode);
                            }
                            semanticOutput = 0;
                        }
                        if (!connectNodes(semanticNode, semanticNode.outputs?.[semanticOutput]?.name, clip, config.input)) throw new Error("无法连接片段参考视频语义");
                    }
                    let trackSource = clip;
                    if (clipData.cached_result && clipData.type === "action") {
                        const loader = mediaNode(clipData.cached_result.asset);
                        const result = this.createNode(TYPES.actionResult, [anchor[0] - 900, anchor[1]], { result_version: Number(clipData.cached_result.version) || 0 });
                        if (!connectNodes(clip, "clip", result, "clip") || !connectNodes(loader, loader.outputs?.[0]?.name, result, "video")) throw new Error("无法连接动作缓存结果");
                        trackSource = result;
                    }
                    if (!this.connectAutogrow(trackSource, "clip", track, "clip")) throw new Error("轨道没有可用片段插槽");
                }
                if (!this.connectAutogrow(track, "track", trackList, "track")) throw new Error("轨道数组没有可用插槽");
            }
            swapped = true;
            for (const name of inputNames) {
                const index = slotIndex(timeline, name);
                if (index >= 0 && timeline.inputs[index].link != null) timeline.disconnectInput(index);
            }
            if (!connectNodes(group, "character_group", timeline, "character_group") || !connectNodes(style, "style_card", timeline, "style_card") ||
                !connectNodes(environment, "environment_instance", timeline, "environment") || !connectNodes(trackList, "tracks", timeline, "tracks")) throw new Error("无法替换总时间轴基础数据");
            setWidgetValue(timeline, "duration_seconds", Number(project.project.duration_seconds));
            setWidgetValue(timeline, "prompt_language", project.project.prompt_language === "中文" ? "中文" : "英文");
            timeline.title = String(project.project.name || "MiniMax H3 项目");
            timeline.properties ||= {};
            timeline.properties.minimax_h3_project = {
                id: String(project.project.id || crypto.randomUUID()),
                extensions: structuredClone(project.extensions || {}),
            };
            const removed = abandonedUpstreamNodes(oldUpstream);
            for (const node of removed.reverse()) app.graph.remove(node);
            this.layoutProjectNodes(false);
            app.graph.afterChange?.();
            app.graph.setDirtyCanvas(true, true);
            this.markWorkflowChanged();
            this.selection = null;
            this.render();
            this.setStatus(`已用“${timeline.title}”覆盖当前时间轴：创建 ${created.length} 个节点，移除 ${removed.length} 个旧节点，下游生成连线保持不变`);
        } catch (error) {
            if (swapped) {
                for (const name of inputNames) {
                    const index = slotIndex(timeline, name);
                    if (index >= 0 && timeline.inputs[index].link != null) timeline.disconnectInput(index);
                    const oldRoot = oldRoots.get(name);
                    if (oldRoot) connectNodes(oldRoot, oldRoot.outputs?.[0]?.name, timeline, name);
                }
                setWidgetValue(timeline, "duration_seconds", oldState.duration);
                setWidgetValue(timeline, "prompt_language", oldState.language);
                timeline.title = oldState.title;
                timeline.properties = oldState.properties;
            }
            for (const node of created.reverse()) app.graph.remove(node);
            app.graph.afterChange?.();
            throw error;
        } finally {
            this.pendingCreatedNodes = null;
        }
    }

    markWorkflowChanged() {
        app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
        this.changeVersion += 1;
        this.setSaveStatus(this.autoSave ? "有修改，等待自动保存" : "有未保存修改");
        if (this.autoSave) this.scheduleAutoSave();
    }

    scheduleAutoSave() {
        clearTimeout(this.saveTimer);
        this.saveTimer = setTimeout(() => this.saveWorkflow(false), 1800);
    }

    async saveWorkflow(manual = true) {
        clearTimeout(this.saveTimer);
        this.saveTimer = null;
        const workflow = app.extensionManager?.workflow?.activeWorkflow;
        const command = app.extensionManager?.command;
        if (!workflow || !command?.execute) {
            this.setSaveStatus("当前 ComfyUI 前端不支持工作流保存命令");
            return;
        }
        if (!manual && !workflow.isPersisted) {
            this.setSaveStatus("未命名工作流，请先手动保存一次");
            return;
        }
        const version = this.changeVersion;
        this.setSaveStatus("正在保存工作流…");
        try {
            await command.execute(workflow.isPersisted ? "Comfy.SaveWorkflow" : "Comfy.SaveWorkflowAs");
            if (!app.extensionManager?.workflow?.activeWorkflow?.isPersisted) {
                this.setSaveStatus("保存未完成，请重新点击保存");
                return;
            }
            if (version === this.changeVersion) {
                this.changeVersion = 0;
                this.setSaveStatus(`已保存 · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`);
            } else {
                this.setSaveStatus("保存期间又有修改，等待再次保存");
                if (this.autoSave) this.scheduleAutoSave();
            }
        } catch (error) {
            this.setSaveStatus(`保存失败：${error?.message || error}`);
        }
    }

    render() {
        const timelines = timelineNodes();
        const select = $("[data-role=timeline]", this.root);
        select.innerHTML = timelines.map((node, index) => `<option value="${node.id}" ${node.id === this.timelineId ? "selected" : ""}>${escapeHtml(node.title || `总时间轴 ${index + 1}`)} · #${node.id}</option>`).join("");
        const timeline = this.currentTimeline();
        this.data = timeline ? collectTimeline(timeline) : null;
        if (this.data) this.playhead = Math.min(this.playhead, this.data.duration);
        $("[data-role=duration]", this.root).value = this.data?.duration ?? "";
        $("[data-role=duration]", this.root).disabled = !this.data;
        $("[data-role=prompt-language]", this.root).value = this.data?.promptLanguage || "英文";
        $("[data-role=prompt-language]", this.root).disabled = !this.data;
        this.renderLibrary();
        this.renderTracks();
        this.renderInspector();
        this.updateTime();
    }

    sceneItems() {
        if (!this.data) return [];
        const items = [];
        const group = this.data.characterGroup;
        if (group) {
            numberedInputs(group, ["actors", "actor"]).forEach((actor, index) => {
                items.push({ type: "人物", title: actorName(actor, `人物 ${index + 1}`), node: actor, card: findCardForActor(actor), color: "actor" });
            });
        }
        const environment = this.data.environment;
        if (environment) items.push({ type: "环境", title: environmentName(environment, "环境"), node: environment, card: findCardForEnvironment(environment), color: "environment" });
        if (this.data.style) items.push({ type: "风格", title: shortText(widgetValue(this.data.style, "style"), 22) || "全局风格", node: this.data.style, color: "style" });
        return items;
    }

    renderLibrary() {
        const host = $("[data-role=library-scene]", this.root);
        this.renderResourceLibrary($("[data-role=library-resources]", this.root));
        this.renderReferenceAssets($("[data-role=library-media]", this.root));
        if (!this.data) {
            host.innerHTML = `<div class="h3te-empty">工作流中没有 MiniMax H3 总时间轴节点。</div>`;
            return;
        }
        const items = this.sceneItems();
        host.innerHTML = items.length ? items.map((item, index) => `
            <div class="h3te-scene-resource"><button class="h3te-resource ${item.color}" data-resource="${index}"><span>${escapeHtml(item.type)}</span><strong>${escapeHtml(item.title)}</strong></button><button class="h3te-delete-resource" data-delete-scene-node="${item.node.id}" title="删除${escapeAttribute(item.type)}">×</button></div>`).join("") : `<div class="h3te-empty">总时间轴尚未连接场景资源。</div>`;
        this.resources = items;
    }

    async loadResourceLibrary() {
        try {
            const response = await api.fetchApi("/minimax-h3/resources");
            const body = await response.json();
            if (!response.ok || !body.success) throw new Error(body.error || "读取资源库失败");
            this.library = body.library;
            this.renderLibrary();
            if (this.selection) this.renderInspector();
        } catch (error) {
            this.library = { error: String(error?.message || error) };
            this.renderLibrary();
        }
    }

    async loadReferenceAssets() {
        try {
            const response = await api.fetchApi("/minimax-h3/reference-assets");
            const body = await response.json();
            if (!response.ok || !body.success) throw new Error(body.error || "读取参考视频资产失败");
            this.referenceAssets = body.assets;
        } catch (error) {
            this.referenceAssets = { error: String(error?.message || error) };
        }
        this.renderReferenceAssets($("[data-role=library-media]", this.root));
    }

    referenceAssetPath(asset) {
        const file = asset?.file;
        return file?.subfolder ? `${file.subfolder}/${file.filename}` : file?.filename || "";
    }

    renderReferenceAssets(host) {
        const toolbar = `<div class="h3te-resource-create"><button data-action="import-reference-asset">＋导入并预处理视频</button><button data-action="reload-reference-assets">刷新</button></div>`;
        if (!this.referenceAssets) {
            host.innerHTML = `${toolbar}<div class="h3te-empty">正在读取参考视频资产…</div>`;
            return;
        }
        if (this.referenceAssets.error) {
            host.innerHTML = `${toolbar}<div class="h3te-empty">${escapeHtml(this.referenceAssets.error)}</div>`;
            return;
        }
        const canAttach = this.selection?.kind === "clip" && REFERENCE_SLOTS[nodeType(this.selection.node)];
        const actionSelection = canAttach && nodeType(this.selection.node) === TYPES.action;
        host.innerHTML = `${toolbar}<div class="h3te-resource-count">${this.referenceAssets.length} 个已预处理资产 · 统一为 24 FPS</div>${this.referenceAssets.length ? this.referenceAssets.map((asset) => `
            <article class="h3te-library-card media">
                <video class="h3te-video-preview" controls preload="metadata" src="${escapeAttribute(this.referenceVideoUrl(this.referenceAssetPath(asset)))}"></video>
                <div><span>参考视频资产</span><strong>${escapeHtml(asset.display_name)}</strong><small>${Number(asset.duration).toFixed(2)} 秒 · ${asset.width}×${asset.height} · ${asset.preprocess?.fps || 24} FPS · ${(asset.people || []).length} 个人物声明</small>${asset.description ? `<small>${escapeHtml(asset.description)}</small>` : ""}</div>
                <div class="h3te-library-actions">
                    ${canAttach && !actionSelection ? `<button data-use-reference-asset="${asset.id}">用于当前片段</button>` : ""}
                    ${actionSelection ? ((asset.people || []).length ? (asset.people || []).map((person) => `<button data-use-reference-person="${asset.id}:${person.id}">${escapeHtml(person.id)} → 当前人物</button>`).join("") : `<button disabled>请先声明源人物</button>`) : ""}
                    <button data-edit-reference-asset="${asset.id}">属性</button>
                    <button data-instantiate-reference-asset="${asset.id}">创建四路输出</button>
                    <button class="danger" data-delete-reference-asset="${asset.id}">移出资产库</button>
                </div>
            </article>`).join("") : `<div class="h3te-empty">尚未导入参考视频。导入时会完成截取、24 FPS 重采样和音频标准化封装。</div>`}`;
    }

    editReferenceAsset(assetId) {
        const asset = Array.isArray(this.referenceAssets) ? this.referenceAssets.find((item) => item.id === assetId) : null;
        if (!asset) return;
        this.selection = { kind: "video_asset", asset: structuredClone(asset), title: asset.display_name };
        this.renderInspector();
    }

    referenceAssetForm(asset) {
        return `<section class="h3te-form-section">
            <div class="h3te-form-title"><div><small>MiniMax H3 Reference Asset</small><strong>${escapeHtml(asset.display_name)}</strong></div></div>
            <video class="h3te-video-preview" controls preload="metadata" src="${escapeAttribute(this.referenceVideoUrl(this.referenceAssetPath(asset)))}"></video>
            <label class="h3te-field"><span>资产名称</span><input type="text" value="${escapeAttribute(asset.display_name)}" data-reference-asset-field="display_name"></label>
            <label class="h3te-field"><span>视频内容说明</span><textarea rows="4" data-reference-asset-field="description">${escapeHtml(asset.description || "")}</textarea></label>
            <div class="h3te-form-title"><div><small>按画面中的稳定特征区分人物</small><strong>源视频人物声明</strong></div><button data-add-reference-person>＋人物</button></div>
            ${(asset.people || []).map((person, index) => `<div class="h3te-reference-person-row">
                <label class="h3te-field"><span>人物编号</span><input type="text" value="${escapeAttribute(person.id)}" data-reference-person-field="id" data-reference-person-index="${index}"></label>
                <label class="h3te-field"><span>识别描述</span><textarea rows="3" data-reference-person-field="description" data-reference-person-index="${index}">${escapeHtml(person.description)}</textarea></label>
                <button class="danger" data-remove-reference-person="${index}">删除人物声明</button>
            </div>`).join("") || `<div class="h3te-empty">尚未声明人物。多人视频应为每位表演者添加稳定、可见、互不混淆的识别描述。</div>`}
            <button class="primary" data-save-reference-asset="${asset.id}">保存视频资产属性</button>
        </section>`;
    }

    addReferencePerson() {
        const asset = this.selection?.kind === "video_asset" ? this.selection.asset : null;
        if (!asset) return;
        asset.people ||= [];
        const used = new Set(asset.people.map((person) => person.id));
        let index = 1;
        while (used.has(`person_${index}`)) index++;
        asset.people.push({ id: `person_${index}`, description: "" });
        this.renderInspector();
    }

    removeReferencePerson(index) {
        const asset = this.selection?.kind === "video_asset" ? this.selection.asset : null;
        if (!asset?.people?.[index]) return;
        asset.people.splice(index, 1);
        this.renderInspector();
    }

    async saveReferenceAsset(assetId) {
        const asset = this.selection?.kind === "video_asset" && this.selection.asset.id === assetId ? this.selection.asset : null;
        if (!asset) return;
        try {
            const response = await api.fetchApi(`/minimax-h3/reference-assets/${encodeURIComponent(assetId)}`, {
                method: "PUT", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ display_name: asset.display_name, description: asset.description, people: asset.people }),
            });
            const result = await response.json();
            if (!response.ok || !result.success) throw new Error(result.error || "保存参考视频属性失败");
            this.selection.asset = structuredClone(result.asset);
            await this.loadReferenceAssets();
            this.renderInspector();
            this.setStatus("参考视频内容与人物声明已保存");
        } catch (error) {
            this.setStatus(`保存失败：${error?.message || error}`);
        }
    }

    renderReferenceAssetModal() {
        const host = $("[data-role=reference-asset-modal]", this.root);
        if (!this.referenceAssetModal) {
            host.classList.remove("is-open");
            host.innerHTML = "";
            return;
        }
        host.classList.add("is-open");
        host.innerHTML = `<form class="h3te-resource-form" data-reference-asset-form>
            <div class="h3te-modal-title"><div><small>一次导入，重复使用</small><strong>导入参考视频资产</strong></div><button type="button" data-action="close-reference-asset-modal">×</button></div>
            <div class="h3te-modal-fields">
                <label class="h3te-field"><span>资产名称</span><input name="name" type="text" placeholder="例如：双人舞蹈动作 01"></label>
                <label class="h3te-field"><span>视频文件</span><input name="video" type="file" accept="video/*" required></label>
                <label class="h3te-field"><span>截取开始（秒）</span><input name="trim_start" type="number" min="0" step="0.01" value="0"></label>
                <label class="h3te-field"><span>截取结束（秒，0 为原片结尾）</span><input name="trim_end" type="number" min="0" step="0.01" value="0"></label>
                <small>导入时会生成独立 MP4 资产，按 24 FPS 预处理并保留原宽高比。长视频可导入一次，再由不同轨道片段分别截取使用。</small>
                <div class="h3te-form-error" data-role="reference-asset-error"></div>
            </div>
            <div class="h3te-modal-actions"><button type="button" data-action="close-reference-asset-modal">取消</button><button class="primary" type="submit">导入并预处理</button></div>
        </form>`;
        $("[data-reference-asset-form]", host).addEventListener("submit", (event) => this.importReferenceAsset(event));
    }

    async importReferenceAsset(event) {
        event.preventDefault();
        const form = event.currentTarget;
        const submit = form.querySelector("button[type=submit]");
        const errorHost = $("[data-role=reference-asset-error]", form);
        submit.disabled = true;
        this.setStatus("正在导入并预处理参考视频…");
        try {
            const body = new FormData();
            body.append("video", form.elements.video.files[0]);
            body.append("name", form.elements.name.value.trim());
            body.append("trim_start", form.elements.trim_start.value || "0");
            body.append("trim_end", form.elements.trim_end.value || "0");
            const response = await api.fetchApi("/minimax-h3/reference-assets", { method: "POST", body });
            const result = await response.json();
            if (!response.ok || !result.success) throw new Error(result.error || "参考视频资产导入失败");
            this.referenceAssetModal = false;
            this.renderReferenceAssetModal();
            await this.loadReferenceAssets();
            this.setStatus(`${result.asset.display_name} 已完成预处理，可直接创建人物、镜头、灯光和音频输出`);
        } catch (error) {
            errorHost.textContent = `导入失败：${error?.message || error}`;
            submit.disabled = false;
            this.setStatus(errorHost.textContent);
        }
    }

    instantiateReferenceAsset(assetId, attachToSelection = false, personId = null) {
        const asset = Array.isArray(this.referenceAssets) ? this.referenceAssets.find((item) => item.id === assetId) : null;
        const clip = attachToSelection && this.selection?.kind === "clip" ? this.selection.node : null;
        const config = clip ? REFERENCE_SLOTS[nodeType(clip)] : null;
        if (!asset || (attachToSelection && !config)) return;
        const person = personId ? (asset.people || []).find((item) => item.id === personId) : null;
        if (clip && nodeType(clip) === TYPES.action && !person) {
            this.setStatus("人物动作参考必须先选择视频资产中声明的源人物");
            return;
        }
        const timeline = this.currentTimeline();
        const base = clip?.pos || timeline?.pos || app.canvas.graph_mouse || [700, 300];
        const standaloneOffset = clip ? 0 : this.createdOffset++ * 170;
        const anchor = [base[0], base[1] + standaloneOffset];
        const created = [];
        this.pendingCreatedNodes = created;
        app.graph.beforeChange?.();
        try {
            const previousReference = clip ? linkedNode(clip, config.input) : null;
            const previousSplit = nodeType(previousReference) === TYPES.videoPerson ? linkedNode(previousReference, "motion_reference") : previousReference;
            const previousLoader = previousSplit ? linkedNode(previousSplit, "video") : null;
            const loader = this.createNode(TYPES.loadVideo, [anchor[0] - 760, anchor[1]], { file: this.referenceAssetPath(asset) });
            const clipDuration = clip ? Math.max(0.05, Number(widgetValue(clip, "end_time", 1)) - Number(widgetValue(clip, "start_time", 0))) : 0;
            const reference = this.createNode(TYPES.motionReference, [anchor[0] - 460, anchor[1]], {
                trim_start: 0, trim_end: clip ? Math.min(Number(asset.duration) || clipDuration, clipDuration) : 0,
            });
            if (!connectNodes(loader, loader.outputs?.[0]?.name, reference, "video")) throw new Error("无法连接参考视频资产");
            let semanticNode = reference;
            if (person) {
                semanticNode = this.createNode(TYPES.videoPerson, [anchor[0] - 220, anchor[1]], {
                    person_id: person.id, person_description: person.description,
                });
                if (!connectNodes(reference, reference.outputs?.[0]?.name, semanticNode, "motion_reference")) throw new Error("无法连接参考视频人物解释");
            }
            if (clip) {
                const input = slotIndex(clip, config.input);
                const output = person ? 0 : config.output;
                if (input < 0 || !semanticNode.outputs?.[output]) throw new Error("当前片段不支持该参考语义");
                semanticNode.connect(output, clip, input);
            }
            loader.properties ||= {};
            reference.properties ||= {};
            loader.properties.minimax_h3_reference_asset = asset.id;
            reference.properties.minimax_h3_reference_asset = asset.id;
            if (person) {
                semanticNode.properties ||= {};
                semanticNode.properties.minimax_h3_reference_asset = asset.id;
                semanticNode.properties.minimax_h3_reference_person = person.id;
            }
            if (clip) {
                if (isManagedMediaNode(previousReference) && outputTargets(previousReference).length === 0) app.graph.remove(previousReference);
                if (previousSplit !== previousReference && isManagedMediaNode(previousSplit) && outputTargets(previousSplit).length === 0) app.graph.remove(previousSplit);
                if (isManagedMediaNode(previousLoader) && outputTargets(previousLoader).length === 0) app.graph.remove(previousLoader);
            }
            if (timeline) this.layoutProjectNodes(false);
            app.graph.afterChange?.();
            this.markWorkflowChanged();
            this.render();
            this.setStatus(clip ? `${asset.display_name}${person ? ` 的 ${person.id}` : ""} 已连接到当前片段` : `${asset.display_name} 已创建四路语义输出节点`);
        } catch (error) {
            for (const node of created.reverse()) app.graph.remove(node);
            app.graph.afterChange?.();
            this.setStatus(`创建参考资产节点失败：${error?.message || error}`);
        } finally {
            this.pendingCreatedNodes = null;
        }
    }

    async deleteReferenceAsset(assetId) {
        const asset = Array.isArray(this.referenceAssets) ? this.referenceAssets.find((item) => item.id === assetId) : null;
        if (!asset || !await this.confirmDelete(`将“${asset.display_name}”移出资产库？已创建的工作流节点和媒体文件会保留。`)) return;
        const response = await api.fetchApi(`/minimax-h3/reference-assets/${encodeURIComponent(assetId)}`, { method: "DELETE" });
        const result = await response.json();
        if (!response.ok || !result.success) {
            this.setStatus(`删除失败：${result.error || response.statusText}`);
            return;
        }
        await this.loadReferenceAssets();
        this.setStatus("参考视频已移出资产库，现有工作流连接保持不变");
    }

    async saveResourceLibrary(nextLibrary) {
        const response = await api.fetchApi("/minimax-h3/resources", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ expected_revision: this.library?.revision ?? 0, library: nextLibrary }),
        });
        const body = await response.json();
        if (!response.ok || !body.success) throw new Error(body.error || "保存资源库失败");
        this.library = body.library;
        this.renderLibrary();
        return body.library;
    }

    renderResourceLibrary(host) {
        const toolbar = `<div class="h3te-resource-create"><button data-create-resource="characters">＋人物</button><button data-create-resource="environments">＋环境</button><button data-create-resource="styles">＋风格</button></div>
            <input class="h3te-resource-search" data-role="resource-search" value="${escapeAttribute(this.resourceFilter)}" placeholder="搜索名称或标签">`;
        if (!this.library) {
            host.innerHTML = `${toolbar}<div class="h3te-empty">正在读取资源库…</div>`;
            return;
        }
        if (this.library.error) {
            host.innerHTML = `${toolbar}<div class="h3te-empty">${escapeHtml(this.library.error)}<br><button data-action="reload-library">重新读取</button></div>`;
            return;
        }
        const query = this.resourceFilter.trim().toLowerCase();
        const entries = [];
        for (const kind of Object.keys(RESOURCE_NAMES)) {
            for (const resource of this.library[kind] || []) {
                const searchable = `${resource.display_name} ${(resource.tags || []).join(" ")}`.toLowerCase();
                if (!query || searchable.includes(query)) entries.push({ kind, resource });
            }
        }
        const extensionEntries = (this.library.extension_cards || []).filter((resource) => {
            const searchable = `${resource.display_name} ${resource.kind} ${(resource.tags || []).join(" ")}`.toLowerCase();
            return !query || searchable.includes(query);
        });
        host.innerHTML = `${toolbar}<div class="h3te-resource-count">${entries.length} 个核心资源 · ${extensionEntries.length} 个扩展资源 · 版本 ${this.library.revision}</div>${entries.length ? entries.map(({ kind, resource }) => `
            <article class="h3te-library-card ${kind}">
                <div><span>${RESOURCE_NAMES[kind]}</span><strong>${escapeHtml(resource.display_name)}</strong><small>${escapeHtml((resource.tags || []).join(" · ") || "无标签")}</small></div>
                <div class="h3te-library-actions"><button data-instantiate-resource="${kind}:${resource.id}">实例化</button><button data-edit-resource="${kind}:${resource.id}">编辑</button><button class="danger" data-delete-resource="${kind}:${resource.id}">删除</button></div>
            </article>`).join("") : `<div class="h3te-empty">资源库中还没有核心卡片。点击上方按钮自主创建。</div>`}${extensionEntries.map((resource) => `
            <article class="h3te-library-card extensions">
                <div><span>扩展 · ${escapeHtml(resource.kind)}</span><strong>${escapeHtml(resource.display_name)}</strong><small>由对应扩展节点包管理</small></div>
            </article>`).join("")}`;
    }

    newResource(kind) {
        const common = { id: crypto.randomUUID(), revision: 1, display_name: "", tags: [], reference_image: null };
        if (kind === "characters") return { ...common, card: { name: "", description: "", style_priority: "global", character_style: "" }, instance_defaults: { position_override: "", pose_override: "", emotion_override: "", appearance_override: "" } };
        if (kind === "environments") return { ...common, card: { name: "", location: "", default_background: "" }, instance_defaults: { location_override: "", time_weather_override: "", background_override: "", atmosphere_override: "" } };
        return { ...common, card: { style: "", rendering: "", color_palette: "", texture: "", reference_usage: "" } };
    }

    findLibraryResource(kind, id) {
        return this.library?.[kind]?.find((resource) => resource.id === id);
    }

    openResourceModal(kind, id = null) {
        const source = id ? this.findLibraryResource(kind, id) : this.newResource(kind);
        if (!source) return;
        this.resourceModal = { kind, editing: Boolean(id), resource: structuredClone(source) };
        this.renderResourceModal();
    }

    imageReferenceFromCard(card) {
        const source = linkedNode(card, "reference_image");
        const value = String(widgetValue(source, "image", "")).replaceAll("\\", "/").replace(/\s+\[(?:input|output|temp)\]$/, "");
        if (nodeType(source) === TYPES.loadImage && value) addWidgetOption(source, "image", value);
        const parts = value.split("/").filter(Boolean);
        const filename = parts.pop();
        if (!filename || parts.includes("..") || filename === ".." || value.includes(":")) return null;
        return { type: "input", subfolder: parts.join("/"), filename };
    }

    saveActorAsResource(actor) {
        if (!this.library || this.library.error) {
            this.setStatus("人物卡资源库尚未读取完成");
            return;
        }
        const card = findCardForActor(actor);
        if (!card) {
            this.setStatus("人物实例没有连接人物卡，无法保存");
            return;
        }
        const resource = this.newResource("characters");
        resource.display_name = widgetValue(card, "name", "未命名人物");
        for (const name of ["name", "description", "style_priority", "character_style"]) resource.card[name] = widgetValue(card, name, resource.card[name]);
        for (const name of ["position_override", "pose_override", "emotion_override", "appearance_override"]) resource.instance_defaults[name] = widgetValue(actor, name, "");
        resource.reference_image = this.imageReferenceFromCard(card);
        this.resourceModal = { kind: "characters", editing: false, resource };
        this.renderResourceModal();
    }

    characterCardPicker(card) {
        const resources = Array.isArray(this.library?.characters) ? this.library.characters : [];
        const currentId = card.properties?.minimax_h3_resource?.resource_kind === "characters" ? card.properties.minimax_h3_resource.resource_id : "";
        const animaTypes = globalThis.LiteGraph?.registered_node_types || {};
        const hasAnimaLibrary = Boolean(animaTypes.AnimaCharacterTagSelector || animaTypes.AnimaCharacterTagSelectorPlus);
        return `<div class="h3te-card-picker">
            <label class="h3te-field"><span>选择已有人物卡</span><select data-character-card-select="${card.id}" ${resources.length ? "" : "disabled"}>
                <option value="">${resources.length ? "请选择资源库人物卡" : "人物卡资源库为空"}</option>
                ${resources.map((resource) => `<option value="${escapeAttribute(resource.id)}" ${resource.id === currentId ? "selected" : ""}>${escapeHtml(resource.display_name)}</option>`).join("")}
            </select></label>
            <button data-apply-character-card="${card.id}" ${resources.length ? "" : "disabled"}>应用到当前人物卡</button>
            <button data-select-anima-character="${card.id}" ${hasAnimaLibrary ? "" : "disabled"}>从 Anima 角色库选择</button>
            <small>只替换固定外观、人物风格和参考图；人物实例状态与动作轨道保持不变。</small>
            ${hasAnimaLibrary ? "<small>Anima 角色将转换为固定外观描述，不会写入动作或人物实例状态。</small>" : "<small>未检测到 Comfyui-Anima-Tools，安装并刷新前端后可使用其人物库。</small>"}
        </div>`;
    }

    cardImagePicker(card) {
        const labels = { [TYPES.character]: "人物", [TYPES.environment]: "环境", [TYPES.visual]: "风格" };
        const label = labels[nodeType(card)];
        if (!label) return "";
        const reference = this.imageReferenceFromCard(card);
        const path = reference ? (reference.subfolder ? `${reference.subfolder}/${reference.filename}` : reference.filename) : "";
        const preview = reference ? `<img class="h3te-image-preview" src="${escapeAttribute(this.referenceImageUrl(reference))}" alt="人物参考图预览">` : "";
        return `<div class="h3te-card-picker">
            ${preview}
            <label class="h3te-field"><span>参考图片</span><input type="file" accept="image/*" data-card-image-file="${card.id}"></label>
            <button data-attach-card-image="${card.id}">${reference ? `替换${label}参考图片` : `上传并连接${label}参考图片`}</button>
            <small>${path ? `当前：${escapeHtml(path)}` : "尚未连接参考图片；上传后会自动创建原生加载图像节点。"}</small>
        </div>`;
    }

    async attachCardImage(cardId) {
        const card = app.graph.getNodeById(Number(cardId));
        const input = $(`[data-card-image-file="${cardId}"]`, this.root);
        const file = input?.files?.[0];
        if (!card || ![TYPES.character, TYPES.environment, TYPES.visual].includes(nodeType(card)) || !file) {
            this.setStatus("请先选择一张参考图片");
            return;
        }
        this.setStatus("正在上传并连接参考图片…");
        try {
            const reference = await this.uploadResourceImage(file);
            const path = reference.subfolder ? `${reference.subfolder}/${reference.filename}` : reference.filename;
            const created = [];
            this.pendingCreatedNodes = created;
            app.graph.beforeChange?.();
            try {
                const loader = linkedNode(card, "reference_image");
                if (nodeType(loader) === TYPES.loadImage) {
                    setImageLoaderValue(loader, path);
                } else {
                    this.addReferenceImage({ reference_image: reference }, card, [card.pos[0] - 300, card.pos[1]]);
                }
                this.layoutProjectNodes(false);
                app.graph.afterChange?.();
                this.markWorkflowChanged();
                this.data = collectTimeline(this.currentTimeline());
                this.renderInspector();
                this.setStatus("参考图片已上传并连接到当前卡片");
            } catch (error) {
                for (const node of created.reverse()) app.graph.remove(node);
                app.graph.afterChange?.();
                throw error;
            } finally {
                this.pendingCreatedNodes = null;
            }
        } catch (error) {
            this.setStatus(`参考图片处理失败：${error?.message || error}`);
        }
    }

    animaCharacterFromPrompt(prompt) {
        const value = String(prompt || "").trim().toLowerCase();
        const official = window.characterOfficialData || {};
        return (window.characterData || []).map((item) => {
            const key = `${item.name || ""}||${item.copyright || ""}`.toLowerCase();
            const data = official[key] || {};
            const trigger = String(data.trigger || (item.copyright ? `${item.name}, ${item.copyright}` : item.name) || "").trim();
            return { item, data, trigger };
        }).filter(({ trigger }) => trigger && (value === trigger.toLowerCase() || value.startsWith(`${trigger.toLowerCase()},`)))
            .sort((a, b) => b.trigger.length - a.trigger.length)[0] || null;
    }

    animaCharacterDescription(item, official) {
        const readable = (value) => String(value || "").replaceAll("_", " ").replace(/\s+/g, " ").trim();
        const name = readable(item.name);
        const copyright = readable(item.copyright);
        const fallback = [item.gender, item.hair ? `${item.hair} hair` : "", item.eye ? `${item.eye} eyes` : ""].filter(Boolean);
        const traits = (Array.isArray(official.tags) ? official.tags : fallback).map(readable).filter(Boolean).slice(0, 16);
        const source = copyright ? `${name} from ${copyright}` : name;
        return traits.length ? `${source}. Fixed visual appearance: ${traits.join(", ")}.` : `${source}. Preserve the character's fixed visual appearance.`;
    }

    async importAnimaCharacterImage(name, copyright) {
        const response = await api.fetchApi("/minimax-h3/anima-character-image", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, copyright }),
        });
        const result = await response.json();
        if (!response.ok || !result.success) throw new Error(result.error || "Anima 人物参考图导入失败");
        return result;
    }

    async selectAnimaCharacter(cardId) {
        const card = app.graph.getNodeById(Number(cardId));
        const animaTypes = globalThis.LiteGraph?.registered_node_types || {};
        const selectorType = animaTypes.AnimaCharacterTagSelector ? "AnimaCharacterTagSelector" : animaTypes.AnimaCharacterTagSelectorPlus ? "AnimaCharacterTagSelectorPlus" : "";
        if (!card || nodeType(card) !== TYPES.character || !selectorType) {
            this.setStatus("未检测到 Comfyui-Anima-Tools 人物选择库");
            return;
        }
        const selector = globalThis.LiteGraph.createNode(selectorType);
        const tags = widget(selector, "character_tags");
        const button = selector?.widgets?.find((item) => item.type === "button" && typeof item.callback === "function");
        if (!tags || !button) {
            this.setStatus("Anima 人物选择器未正确加载，请刷新 ComfyUI 前端");
            return;
        }

        const selectedPrompt = await new Promise((resolve) => {
            let finished = false;
            let overlay = null;
            let observer = null;
            let waitForOverlay = null;
            let loadTimeout = null;
            const finish = (value) => {
                if (finished) return;
                finished = true;
                clearInterval(waitForOverlay);
                clearTimeout(loadTimeout);
                observer?.disconnect();
                resolve(value || "");
            };
            const originalCallback = tags.callback;
            tags.callback = (value) => {
                originalCallback?.(value);
                finish(value);
            };
            waitForOverlay = setInterval(() => {
                overlay = document.getElementById("anima-char-selector-overlay");
                if (!overlay) return;
                clearInterval(waitForOverlay);
                clearTimeout(loadTimeout);
                const editorZIndex = Number.parseInt(getComputedStyle(this.root).zIndex, 10) || 100000;
                overlay.style.zIndex = String(editorZIndex + 2);
                observer = new MutationObserver(() => {
                    if (!document.body.contains(overlay)) finish(tags.value);
                });
                observer.observe(document.body, { childList: true });
            }, 50);
            loadTimeout = setTimeout(() => finish(""), 15000);
            Promise.resolve(button.callback()).catch(() => finish(""));
        });
        if (!selectedPrompt) {
            this.setStatus("已取消 Anima 人物选择");
            return;
        }
        const selected = this.animaCharacterFromPrompt(selectedPrompt);
        if (!selected) {
            this.setStatus("无法识别 Anima 角色，请使用“应用触发词”或“应用触发词 + 标签”确认选择");
            return;
        }

        this.setStatus(`正在导入“${selected.item.name}”的 Anima 参考图…`);
        let importedImage = null;
        let imageError = null;
        try {
            importedImage = await this.importAnimaCharacterImage(selected.item.name, selected.item.copyright || "");
        } catch (error) {
            imageError = error;
        }

        const created = [];
        this.pendingCreatedNodes = created;
        app.graph.beforeChange?.();
        try {
            setWidgetValue(card, "name", String(selected.item.name || "").replaceAll("_", " "));
            setWidgetValue(card, "description", this.animaCharacterDescription(selected.item, selected.data));
            if (importedImage?.reference_image) {
                const reference = importedImage.reference_image;
                const path = reference.subfolder ? `${reference.subfolder}/${reference.filename}` : reference.filename;
                const loader = linkedNode(card, "reference_image");
                if (nodeType(loader) === TYPES.loadImage) {
                    setImageLoaderValue(loader, path);
                } else {
                    this.addReferenceImage({ reference_image: reference }, card, [card.pos[0] - 300, card.pos[1]]);
                }
            }
            card.properties ||= {};
            card.properties.minimax_h3_anima_character = {
                name: selected.item.name,
                copyright: selected.item.copyright || "",
                trigger: selected.trigger,
                source_url: importedImage?.source_url || "",
                reference_image: importedImage?.reference_image || null,
            };
            delete card.properties.minimax_h3_resource;
            this.layoutProjectNodes(false);
            app.graph.afterChange?.();
            this.markWorkflowChanged();
            this.data = collectTimeline(this.currentTimeline());
            this.renderLibrary();
            this.renderInspector();
            this.setStatus(imageError
                ? `已载入“${selected.item.name}”的文字外观，但参考图导入失败：${imageError.message || imageError}`
                : `已载入“${selected.item.name}”并将 Anima 图片连接为人物参考图`);
        } catch (error) {
            for (const node of created.reverse()) app.graph.remove(node);
            app.graph.afterChange?.();
            this.setStatus(`应用 Anima 人物失败：${error?.message || error}`);
        } finally {
            this.pendingCreatedNodes = null;
        }
    }

    applyCharacterCard(cardId) {
        const card = app.graph.getNodeById(Number(cardId));
        const select = $(`[data-character-card-select="${cardId}"]`, this.root);
        const resource = this.findLibraryResource("characters", select?.value);
        if (!card || nodeType(card) !== TYPES.character || !resource) {
            this.setStatus("请先选择一张已有人物卡");
            return;
        }
        const created = [];
        this.pendingCreatedNodes = created;
        app.graph.beforeChange?.();
        try {
            const previousLoader = linkedNode(card, "reference_image");
            if (resource.reference_image) {
                const path = resource.reference_image.subfolder ? `${resource.reference_image.subfolder}/${resource.reference_image.filename}` : resource.reference_image.filename;
                if (nodeType(previousLoader) === TYPES.loadImage) setImageLoaderValue(previousLoader, path);
                else this.addReferenceImage(resource, card, [card.pos[0] - 300, card.pos[1]]);
            } else {
                const input = slotIndex(card, "reference_image");
                if (input >= 0 && card.inputs[input].link != null) card.disconnectInput(input);
                if (isManagedMediaNode(previousLoader) && outputTargets(previousLoader).length === 0) app.graph.remove(previousLoader);
            }
            for (const name of ["name", "description", "style_priority", "character_style"]) setWidgetValue(card, name, resource.card[name]);
            card.properties ||= {};
            card.properties.minimax_h3_resource = {
                library_id: this.library.library_id,
                resource_id: resource.id,
                resource_kind: "characters",
                resource_revision: resource.revision,
            };
            this.layoutProjectNodes(false);
            app.graph.afterChange?.();
            this.markWorkflowChanged();
            this.data = collectTimeline(this.currentTimeline());
            this.renderLibrary();
            this.renderInspector();
            this.setStatus(`已将人物卡切换为“${resource.display_name}”，人物实例和动作保持不变`);
        } catch (error) {
            for (const node of created.reverse()) app.graph.remove(node);
            app.graph.afterChange?.();
            this.setStatus(`应用人物卡失败：${error?.message || error}`);
        } finally {
            this.pendingCreatedNodes = null;
        }
    }

    renderResourceModal() {
        const host = $("[data-role=resource-modal]", this.root);
        if (!this.resourceModal) {
            host.classList.remove("is-open");
            host.innerHTML = "";
            return;
        }
        const { kind, editing, resource } = this.resourceModal;
        host.classList.add("is-open");
        host.innerHTML = `<form class="h3te-resource-form" data-resource-form>
            <div class="h3te-modal-title"><div><small>${editing ? "编辑资源" : "自主创建卡片"}</small><strong>${RESOURCE_NAMES[kind]}卡</strong></div><button type="button" data-action="close-resource-modal">×</button></div>
            <div class="h3te-modal-fields">${RESOURCE_FORMS[kind].map(([path, label, type, values]) => {
                const value = path === "tags" ? (resource.tags || []).join(", ") : pathValue(resource, path);
                const attrs = `data-resource-path="${path}"`;
                if (type === "select") return `<label class="h3te-field"><span>${label}</span><select ${attrs}>${values.map((item) => `<option ${item === value ? "selected" : ""}>${item}</option>`).join("")}</select></label>`;
                if (type === "textarea") return `<label class="h3te-field"><span>${label}</span><textarea rows="3" ${attrs}>${escapeHtml(value)}</textarea></label>`;
                return `<label class="h3te-field"><span>${label}</span><input type="text" value="${escapeAttribute(value)}" ${attrs}></label>`;
            }).join("")}
            <label class="h3te-field"><span>参考图片（可选）</span><input type="file" accept="image/*" data-resource-image><small>${resource.reference_image ? `当前：${escapeHtml(resource.reference_image.subfolder ? `${resource.reference_image.subfolder}/${resource.reference_image.filename}` : resource.reference_image.filename)}` : "保存后实例化时会自动连接原生加载图片节点"}</small></label>
            ${resource.reference_image ? `<label class="h3te-check"><input type="checkbox" data-remove-resource-image> 移除已有参考图</label>` : ""}
            <div class="h3te-form-error" data-role="resource-form-error"></div>
            </div>
            <div class="h3te-modal-actions"><button type="button" data-action="close-resource-modal">取消</button><button class="primary" type="submit">保存卡片</button></div>
        </form>`;
        $("[data-resource-form]", host).addEventListener("submit", (event) => this.submitResourceForm(event));
    }

    async uploadResourceImage(file) {
        const body = new FormData();
        body.append("image", file, file.name);
        body.append("type", "input");
        body.append("subfolder", "minimax_h3/resources");
        body.append("overwrite", "false");
        const response = await api.fetchApi("/upload/image", { method: "POST", body });
        if (!response.ok) throw new Error(await response.text() || "参考图上传失败");
        const uploaded = await response.json();
        return { type: "input", subfolder: uploaded.subfolder || "", filename: uploaded.name || uploaded.filename };
    }

    async submitResourceForm(event) {
        event.preventDefault();
        const form = event.currentTarget;
        const submit = form.querySelector("button[type=submit]");
        submit.disabled = true;
        try {
            const { kind, editing, resource } = this.resourceModal;
            form.querySelectorAll("[data-resource-path]").forEach((field) => {
                const value = field.dataset.resourcePath === "tags" ? field.value.split(/[,，]/).map((item) => item.trim()).filter(Boolean) : field.value.trim();
                setPathValue(resource, field.dataset.resourcePath, value);
            });
            if (!resource.display_name || (kind !== "styles" && !resource.card.name)) throw new Error("资源名称和卡片名称不能为空");
            const image = form.querySelector("[data-resource-image]")?.files?.[0];
            if (form.querySelector("[data-remove-resource-image]")?.checked) resource.reference_image = null;
            if (image) resource.reference_image = await this.uploadResourceImage(image);
            if (editing) resource.revision = (Number(resource.revision) || 1) + 1;
            const next = structuredClone(this.library);
            const index = next[kind].findIndex((item) => item.id === resource.id);
            if (index >= 0) next[kind][index] = resource;
            else next[kind].push(resource);
            await this.saveResourceLibrary(next);
            this.resourceModal = null;
            this.renderResourceModal();
            this.setStatus(`${RESOURCE_NAMES[kind]}卡已保存到 JSON 资源库`);
        } catch (error) {
            const message = `保存失败：${error?.message || error}`;
            const errorHost = $("[data-role=resource-form-error]", form);
            if (errorHost) errorHost.textContent = message;
            this.setStatus(message);
            submit.disabled = false;
        }
    }

    async deleteLibraryResource(kind, id) {
        const resource = this.findLibraryResource(kind, id);
        if (!resource || !await this.confirmDelete(`删除资源“${resource.display_name}”？已实例化的工作流节点不会被删除。`)) return;
        try {
            const next = structuredClone(this.library);
            next[kind] = next[kind].filter((item) => item.id !== id);
            await this.saveResourceLibrary(next);
            this.setStatus("资源已从 JSON 资源库删除，现有工作流节点保持不变");
        } catch (error) {
            this.setStatus(`删除失败：${error?.message || error}`);
        }
    }

    trackOwnedNodes(track) {
        return [track?.node, ...(track?.clips || []).flatMap((clip) => [clip.resultNode, clip.node])].filter(Boolean);
    }

    deleteGraphNodes(nodes, message) {
        const unique = [...new Map(nodes.filter(Boolean).map((node) => [node.id, node])).values()];
        if (!unique.length) return;
        app.graph.beforeChange?.();
        try {
            for (const node of unique.reverse()) app.graph.remove(node);
            this.layoutProjectNodes(false);
        } catch (error) {
            this.setStatus(`删除失败：${error?.message || error}`);
            return;
        } finally {
            app.graph.afterChange?.();
        }
        this.selection = null;
        this.markWorkflowChanged();
        this.data = collectTimeline(this.currentTimeline());
        this.renderLibrary();
        this.renderTracks();
        this.renderInspector();
        this.setStatus(message);
    }

    async cleanupUnusedNodes() {
        const timeline = this.currentTimeline();
        if (!timeline) {
            this.setStatus("请先选择一个总时间轴，作为需要保留的当前工程");
            return;
        }
        const nodes = unusedGraphNodes(timeline);
        if (!nodes.length) {
            this.setStatus("没有发现未使用的 MiniMax H3 工程或资源加载节点");
            return;
        }
        if (!await this.confirmDelete(`删除 ${nodes.length} 个未被当前总时间轴使用的 MiniMax H3 工程或资源加载节点？模型、采样、解码、保存及其他第三方节点不在清理范围内。`)) return;
        this.deleteGraphNodes(nodes, `已删除 ${nodes.length} 个未使用的 MiniMax H3 工程或资源加载节点`);
    }

    async deleteTrack(index) {
        const track = this.data?.tracks?.[Number(index)];
        if (!track || !await this.confirmDelete(`删除“${track.label}”及其 ${track.clips.length} 个片段？此操作不会删除共享的参考视频资产。`)) return;
        this.deleteGraphNodes(this.trackOwnedNodes(track), `已删除轨道“${track.label}”及其片段`);
    }

    async deleteClip(nodeId) {
        const clip = this.data?.tracks?.flatMap((track) => track.clips).find((item) => String(item.node.id) === String(nodeId));
        if (!clip || !await this.confirmDelete(`删除${KIND_NAMES[clip.kind] || "当前"}片段？`)) return;
        this.deleteGraphNodes([clip.resultNode, clip.node], "片段已删除");
    }

    async deleteSceneNode(nodeId) {
        const resource = this.sceneItems().find((item) => String(item.node.id) === String(nodeId));
        if (!resource) return;
        const ownedTracks = this.data.tracks.filter((track) => track.owner === resource.node);
        const nodes = [resource.node, resource.card, ...ownedTracks.flatMap((track) => this.trackOwnedNodes(track))].filter(Boolean);
        const ids = new Set(nodes.map((node) => node.id));
        const loader = resource.card ? linkedNode(resource.card, "reference_image") : linkedNode(resource.node, "reference_image");
        if (nodeType(loader) === TYPES.loadImage && outputTargets(loader).every((targetId) => ids.has(targetId))) nodes.push(loader);
        const trackText = ownedTracks.length ? `、${ownedTracks.length} 条专属轨道及其片段` : "";
        if (!await this.confirmDelete(`删除${resource.type}“${resource.title}”${trackText}？共享参考视频资产不会被删除。`)) return;
        this.deleteGraphNodes(nodes, `已从当前工程删除${resource.type}“${resource.title}”`);
    }

    createNode(type, position, values = {}, resource = null, kind = null) {
        const node = LiteGraph.createNode(type);
        if (!node) throw new Error(`无法创建节点 ${type}`);
        node.pos = position;
        app.graph.add(node);
        this.pendingCreatedNodes?.push(node);
        for (const [name, value] of Object.entries(values || {})) setWidgetValue(node, name, value);
        if (resource) {
            node.properties ||= {};
            node.properties.minimax_h3_resource = {
                library_id: this.library.library_id,
                resource_id: resource.id,
                resource_kind: kind,
                resource_revision: resource.revision,
            };
        }
        return node;
    }

    ensureTimelineNode(timeline, inputName, type, position) {
        const existing = linkedNode(timeline, inputName);
        if (existing) return existing;
        const node = LiteGraph.createNode(type);
        if (!node) throw new Error(`无法创建节点 ${type}`);
        node.pos = position;
        app.graph.add(node);
        this.pendingCreatedNodes?.push(node);
        if (!connectNodes(node, node.outputs?.[0]?.name, timeline, inputName)) throw new Error(`无法连接 ${type}`);
        return node;
    }

    connectAutogrow(origin, outputName, target, prefix) {
        const output = slotIndex(origin, outputName, true);
        const input = freeAutogrowInput(target, prefix);
        if (output < 0 || input < 0) return false;
        origin.connect(output, target, input);
        return true;
    }

    bootstrapProject() {
        const timeline = this.currentTimeline();
        if (!timeline) {
            this.setStatus("请先创建一个 MiniMax H3 总时间轴节点");
            return;
        }
        const created = [];
        this.pendingCreatedNodes = created;
        app.graph.beforeChange?.();
        try {
            const [tx, ty] = timeline.pos;
            const group = this.ensureTimelineNode(timeline, "character_group", "MiniMaxH3CharacterGroup", [tx - 300, ty - 160]);
            let actors = numberedInputs(group, ["actors", "actor"]);
            if (!actors.length) {
                const card = this.createNode(TYPES.character, [tx - 900, ty - 160], { name: "人物 1", description: "", character_style: "" });
                const actor = this.createNode(TYPES.actor, [tx - 600, ty - 160], { actor_id: "actor_1" });
                if (!connectNodes(card, "character_card", actor, "character_card") || !this.connectAutogrow(actor, "actor_instance", group, "actor")) throw new Error("无法创建并连接默认人物");
                actors = [actor];
            }
            if (!linkedNode(timeline, "style_card")) {
                const style = this.createNode(TYPES.visual, [tx - 300, ty - 460], { style: "", rendering: "", color_palette: "", texture: "", reference_usage: "" });
                if (!connectNodes(style, "style_card", timeline, "style_card")) throw new Error("无法连接默认风格卡");
            }
            let environment = linkedNode(timeline, "environment");
            if (!environment) {
                const card = this.createNode(TYPES.environment, [tx - 600, ty + 100], { name: "环境 1", location: "", default_background: "" });
                environment = this.createNode(TYPES.environmentInstance, [tx - 300, ty + 100]);
                if (!connectNodes(card, "environment_card", environment, "environment_card") || !connectNodes(environment, "environment_instance", timeline, "environment")) throw new Error("无法创建并连接默认环境");
            }
            const trackList = this.ensureTimelineNode(timeline, "tracks", TYPES.trackList, [tx - 300, ty + 380]);
            if (!numberedInputs(trackList, ["tracks", "track"]).length) {
                const track = this.createNode(TYPES.actorTrack, [tx - 650, ty + 380]);
                if (!connectNodes(actors[0], "actor_instance", track, "actor") || !this.connectAutogrow(track, "track", trackList, "track")) throw new Error("无法创建默认人物轨道");
            }
            this.layoutProjectNodes(false);
            app.graph.afterChange?.();
            this.markWorkflowChanged();
            this.render();
            this.setStatus(created.length ? `已从总时间轴补全 ${created.length} 个基础节点` : "当前总时间轴的基础节点已经完整");
        } catch (error) {
            for (const node of created.reverse()) app.graph.remove(node);
            app.graph.afterChange?.();
            this.setStatus(`补全工程失败：${error?.message || error}`);
        } finally {
            this.pendingCreatedNodes = null;
        }
    }

    layoutProjectNodes(record = true) {
        const timeline = this.currentTimeline();
        if (!timeline) return false;
        const data = collectTimeline(timeline);
        const [tx, ty] = timeline.pos;
        const placed = new Set();
        let changed = false;
        const place = (node, x, y) => {
            if (!node || placed.has(node.id)) return;
            placed.add(node.id);
            if (node.pos[0] !== x || node.pos[1] !== y) {
                node.pos = [x, y];
                changed = true;
            }
        };
        const unique = (nodes) => [...new Map(nodes.filter(Boolean).map((node) => [node.id, node])).values()];
        const placeColumn = (nodes, x, startY) => {
            let y = startY;
            unique(nodes).forEach((node) => {
                place(node, x, y);
                y += Math.max(150, Number(node.size?.[1]) + 55 || 150);
            });
        };
        if (record) app.graph.beforeChange?.();
        const actors = data.characterGroup ? numberedInputs(data.characterGroup, ["actors", "actor"]) : [];
        const actorCards = actors.map(findCardForActor).filter(Boolean);
        const environmentCard = findCardForEnvironment(data.environment);
        const trackList = linkedNode(timeline, "tracks");
        const mediaNodes = [linkedNode(data.style, "reference_image"), ...actorCards.map((card) => linkedNode(card, "reference_image")), linkedNode(environmentCard, "reference_image")];
        const clipNodes = [];
        data.tracks.forEach((track) => track.clips.forEach((clip) => {
                clipNodes.push(clip.node, clip.resultNode);
                const config = REFERENCE_SLOTS[nodeType(clip.node)];
                const reference = config ? linkedNode(clip.node, config.input) : null;
                const split = nodeType(reference) === TYPES.videoPerson ? linkedNode(reference, "motion_reference") : reference;
                mediaNodes.push(linkedNode(split, "video"), split, reference);
            }));
        const top = ty - 500;
        const gap = 430;
        placeColumn(mediaNodes, tx - gap * 5, top);
        placeColumn([data.style, ...actorCards, environmentCard], tx - gap * 4, top);
        placeColumn([data.characterGroup, ...actors, data.environment], tx - gap * 3, top);
        placeColumn(clipNodes, tx - gap * 2, top);
        placeColumn([...data.tracks.map((track) => track.node), trackList], tx - gap, top);
        place(timeline, tx, ty);
        if (record) app.graph.afterChange?.();
        if (changed) {
            app.graph.setDirtyCanvas(true, true);
            if (record) this.markWorkflowChanged();
        }
        return changed;
    }

    addReferenceImage(resource, card, position) {
        if (!resource.reference_image) return null;
        const loader = LiteGraph.createNode("LoadImage");
        if (!loader) throw new Error("无法创建原生加载图片节点");
        loader.pos = position;
        app.graph.add(loader);
        this.pendingCreatedNodes?.push(loader);
        loader.properties ||= {};
        loader.properties.minimax_h3_managed_media = true;
        const path = resource.reference_image.subfolder ? `${resource.reference_image.subfolder}/${resource.reference_image.filename}` : resource.reference_image.filename;
        setImageLoaderValue(loader, path);
        if (!connectNodes(loader, loader.outputs?.[0]?.name || "IMAGE", card, "reference_image")) throw new Error("无法连接资源参考图");
        return loader;
    }

    async instantiateResource(kind, id) {
        const resource = this.findLibraryResource(kind, id);
        if (!resource) return;
        const timeline = this.currentTimeline();
        const anchor = timeline?.pos || [app.canvas.graph_mouse?.[0] || 700, app.canvas.graph_mouse?.[1] || 300];
        const row = this.createdOffset++;
        const x = anchor[0] - 940;
        const y = anchor[1] + row * 170;
        const createdNodes = [];
        this.pendingCreatedNodes = createdNodes;
        app.graph.beforeChange?.();
        try {
            if (kind === "characters") {
                const card = this.createNode(TYPES.character, [x, y], resource.card, resource, kind);
                const actor = this.createNode(TYPES.actor, [x + 330, y], {
                    ...resource.instance_defaults,
                    actor_id: this.nextActorMacro(timeline),
                }, resource, kind);
                if (!connectNodes(card, "character_card", actor, "character_card")) throw new Error("无法连接人物卡和人物实例");
                this.addReferenceImage(resource, card, [x - 300, y]);
                if (timeline) {
                    const group = this.ensureTimelineNode(timeline, "character_group", "MiniMaxH3CharacterGroup", [x + 650, anchor[1] - 200]);
                    if (!this.connectAutogrow(actor, "actor_instance", group, "actor")) throw new Error("人物组没有可用的自动增长插槽");
                    const trackList = this.ensureTimelineNode(timeline, "tracks", TYPES.trackList, [anchor[0] - 350, anchor[1] + 250]);
                    const track = this.createNode(TYPES.actorTrack, [x + 650, y], {}, resource, kind);
                    if (!connectNodes(actor, "actor_instance", track, "actor")) throw new Error("无法连接人物轨道");
                    if (!this.connectAutogrow(track, "track", trackList, "track")) throw new Error("轨道数组没有可用的自动增长插槽");
                }
            } else if (kind === "environments") {
                const card = this.createNode(TYPES.environment, [x, y], resource.card, resource, kind);
                const instance = this.createNode(TYPES.environmentInstance, [x + 330, y], resource.instance_defaults, resource, kind);
                if (!connectNodes(card, "environment_card", instance, "environment_card")) throw new Error("无法连接环境卡和环境实例");
                this.addReferenceImage(resource, card, [x - 300, y]);
                if (timeline) {
                    if (!linkedNode(timeline, "environment")) connectNodes(instance, "environment_instance", timeline, "environment");
                    const trackList = this.ensureTimelineNode(timeline, "tracks", TYPES.trackList, [anchor[0] - 350, anchor[1] + 250]);
                    const track = this.createNode(TYPES.environmentTrack, [x + 650, y], {}, resource, kind);
                    if (!connectNodes(instance, "environment_instance", track, "environment")) throw new Error("无法连接环境轨道");
                    if (!this.connectAutogrow(track, "track", trackList, "track")) throw new Error("轨道数组没有可用的自动增长插槽");
                }
            } else {
                const card = this.createNode(TYPES.visual, [x, y], resource.card, resource, kind);
                this.addReferenceImage(resource, card, [x - 300, y]);
                if (timeline && !linkedNode(timeline, "style_card")) connectNodes(card, "style_card", timeline, "style_card");
            }
            if (timeline) this.layoutProjectNodes(false);
            app.graph.afterChange?.();
            app.graph.setDirtyCanvas(true, true);
            this.markWorkflowChanged();
            setTimeout(() => this.render(), 100);
            this.setStatus(`${resource.display_name} 已实例化为 ComfyUI 节点`);
        } catch (error) {
            for (const node of createdNodes.reverse()) app.graph.remove(node);
            app.graph.afterChange?.();
            this.setStatus(`实例化失败：${error?.message || error}`);
        } finally {
            this.pendingCreatedNodes = null;
        }
    }

    timelineActors() {
        const group = this.data?.characterGroup;
        return group ? numberedInputs(group, ["actors", "actor"]) : [];
    }

    trackKind(track) {
        if (!track) return "body";
        if (track.type === TYPES.actorTrack) return "body";
        if (track.type === TYPES.environmentTrack) return "environment";
        return track.clips[0]?.kind || "camera";
    }

    openTrackModal(trackIndex = null) {
        if (!this.data) {
            this.setStatus("请先创建并连接总时间轴");
            return;
        }
        const track = trackIndex == null ? null : this.data.tracks[trackIndex];
        const start = Math.min(this.playhead, Math.max(0, this.data.duration - 0.05));
        const end = Math.min(this.data.duration, start + Math.min(1, this.data.duration));
        this.trackModal = { trackIndex, kind: this.trackKind(track), start, end: Math.max(start + 0.05, end) };
        this.renderTrackModal();
    }

    renderTrackModal() {
        const host = $("[data-role=track-modal]", this.root);
        if (!this.trackModal) {
            host.classList.remove("is-open");
            host.innerHTML = "";
            return;
        }
        const existing = this.trackModal.trackIndex != null ? this.data?.tracks[this.trackModal.trackIndex] : null;
        const actors = this.timelineActors();
        const kinds = existing ? (existing.type === TYPES.actorTrack ? ["body", "expression", "gaze"] : [this.trackKind(existing)]) : Object.keys(TRACK_KIND_NAMES);
        host.classList.add("is-open");
        host.innerHTML = `<form class="h3te-resource-form h3te-track-form" data-track-form>
            <div class="h3te-modal-title"><div><small>${existing ? "添加到现有轨道" : "创建 ComfyUI 节点"}</small><strong>${existing ? escapeHtml(existing.label) : "新建轨道与首个片段"}</strong></div><button type="button" data-action="close-track-modal">×</button></div>
            <div class="h3te-modal-fields">
                <label class="h3te-field"><span>片段类型</span><select name="kind" ${existing && existing.type !== TYPES.actorTrack ? "disabled" : ""}>${kinds.map((kind) => `<option value="${kind}">${TRACK_KIND_NAMES[kind]}</option>`).join("")}</select></label>
                <label class="h3te-field" data-track-actor-field><span>人物轨道所属人物</span><select name="actor_id" ${existing || !actors.length ? "disabled" : ""}>${actors.map((actor) => `<option value="${actor.id}" ${existing?.owner === actor ? "selected" : ""}>${escapeHtml(actorName(actor, `人物 #${actor.id}`))}</option>`).join("") || `<option>当前时间轴没有人物</option>`}</select></label>
                <label class="h3te-field"><span>开始时间（秒）</span><input name="start" type="number" min="0" max="${this.data.duration}" step="0.05" value="${this.trackModal.start.toFixed(2)}"></label>
                <label class="h3te-field"><span>结束时间（秒）</span><input name="end" type="number" min="0.05" max="${this.data.duration}" step="0.05" value="${this.trackModal.end.toFixed(2)}"></label>
                <div class="h3te-form-error" data-role="track-form-error"></div>
            </div>
            <div class="h3te-modal-actions"><button type="button" data-action="close-track-modal">取消</button><button class="primary" type="submit">${existing ? "添加片段" : "创建轨道"}</button></div>
        </form>`;
        const form = $("[data-track-form]", host);
        const updateActorField = () => {
            const actorKind = ["body", "expression", "gaze"].includes(form.elements.kind.value);
            const field = $("[data-track-actor-field]", form);
            field.hidden = !actorKind;
            form.elements.actor_id.disabled = !actorKind || Boolean(existing) || !actors.length;
        };
        form.elements.kind.addEventListener("change", updateActorField);
        updateActorField();
        form.addEventListener("submit", (event) => this.submitTrackForm(event));
    }

    clipNodeForKind(kind, position, start, end) {
        const types = { body: TYPES.action, expression: TYPES.action, gaze: TYPES.action, environment: TYPES.environmentAction, camera: TYPES.camera, lighting: TYPES.lighting, audio: TYPES.audio };
        const values = { start_time: start, end_time: end };
        if (["body", "expression", "gaze"].includes(kind)) {
            values.action_type = kind;
            values.content = kind === "body" ? "{actor_1} performs the planned action naturally" : kind === "expression" ? "{actor_1} changes expression naturally" : "{actor_1} shifts gaze naturally";
        }
        return this.createNode(types[kind], position, values);
    }

    async submitTrackForm(event) {
        event.preventDefault();
        const form = event.currentTarget;
        const errorHost = $("[data-role=track-form-error]", form);
        try {
            const existing = this.trackModal.trackIndex == null ? null : this.data.tracks[this.trackModal.trackIndex];
            const kind = existing && existing.type !== TYPES.actorTrack ? this.trackKind(existing) : form.elements.kind.value;
            const start = Number(form.elements.start.value);
            const end = Number(form.elements.end.value);
            if (!(start >= 0 && end > start && end <= this.data.duration + 1e-6)) throw new Error("片段时间必须位于总时间轴内，且结束时间晚于开始时间");
            const timeline = this.currentTimeline();
            const anchor = timeline.pos;
            const created = [];
            this.pendingCreatedNodes = created;
            app.graph.beforeChange?.();
            try {
                const clip = this.clipNodeForKind(kind, [anchor[0] - 920, anchor[1] + 250], start, end);
                let track = existing?.node;
                if (!track) {
                    const trackList = this.ensureTimelineNode(timeline, "tracks", TYPES.trackList, [anchor[0] - 300, anchor[1] + 250]);
                    if (["body", "expression", "gaze"].includes(kind)) {
                        const actor = this.timelineActors().find((item) => String(item.id) === form.elements.actor_id.value);
                        if (!actor) throw new Error("人物轨道必须选择时间轴人物组中的人物");
                        setWidgetValue(clip, "content", String(widgetValue(clip, "content", "")).replace("{actor_1}", `{${widgetValue(actor, "actor_id", "actor_1")}}`));
                        track = this.createNode(TYPES.actorTrack, [anchor[0] - 600, anchor[1] + 250]);
                        if (!connectNodes(actor, "actor_instance", track, "actor")) throw new Error("无法连接人物到人物轨道");
                    } else if (kind === "environment") {
                        if (!this.data.environment) throw new Error("环境轨道需要总时间轴连接环境实例");
                        track = this.createNode(TYPES.environmentTrack, [anchor[0] - 600, anchor[1] + 250]);
                        if (!connectNodes(this.data.environment, "environment_instance", track, "environment")) throw new Error("无法连接环境实例");
                    } else {
                        track = this.createNode(TYPES.systemTrack, [anchor[0] - 600, anchor[1] + 250]);
                    }
                    if (!this.connectAutogrow(track, "track", trackList, "track")) throw new Error("轨道数组没有可用插槽");
                }
                if (!this.connectAutogrow(clip, "clip", track, "clip")) throw new Error("轨道没有可用片段插槽");
                this.layoutProjectNodes(false);
                app.graph.afterChange?.();
                this.trackModal = null;
                this.renderTrackModal();
                this.data = collectTimeline(timeline);
                this.selection = { kind: "clip", node: clip, nodes: [clip], title: `${KIND_NAMES[kind] || kind}片段` };
                this.renderTracks();
                this.renderInspector();
                app.graph.setDirtyCanvas(true, true);
                this.markWorkflowChanged();
                this.setStatus(existing ? "片段已创建并连接到轨道" : "轨道和首个片段已创建并连接到时间轴");
            } catch (error) {
                for (const node of created.reverse()) app.graph.remove(node);
                app.graph.afterChange?.();
                throw error;
            } finally {
                this.pendingCreatedNodes = null;
            }
        } catch (error) {
            errorHost.textContent = `创建失败：${error?.message || error}`;
        }
    }

    renderTracks() {
        const ruler = $("[data-role=ruler]", this.root);
        const host = $("[data-role=tracks]", this.root);
        if (!this.data) {
            ruler.innerHTML = "";
            host.innerHTML = `<div class="h3te-empty h3te-stage-empty">请先在工作流中创建并连接 MiniMax H3 总时间轴。</div>`;
            return;
        }
        const width = Math.max(480, this.data.duration * this.pixelsPerSecond);
        ruler.style.width = `${width}px`;
        ruler.innerHTML = this.rulerHtml(width);
        if (!this.data.tracks.length) {
            host.innerHTML = `<div class="h3te-empty h3te-stage-empty">轨道数组中还没有轨道。请在节点图中连接轨道后刷新。</div>`;
            return;
        }
        this.markConflicts();
        host.innerHTML = this.data.tracks.map((track, index) => `
            <div class="h3te-track-row">
                <div class="h3te-track-head"><button class="h3te-track-label" data-track="${index}"><span>${this.trackIcon(track)}</span><strong>${escapeHtml(track.label)}</strong><small>${track.clips.length} 个片段</small></button><button class="h3te-delete-track" data-delete-track="${index}" title="删除轨道">×</button><button class="h3te-add-clip" data-add-clip="${index}" title="向此轨道添加片段">＋</button></div>
                <div class="h3te-lane-scroll" data-lane-scroll="${index}"><div class="h3te-lane" style="width:${width}px;--grid:${this.pixelsPerSecond}px" data-lane="${index}">
                    ${track.clips.map((clip) => this.clipHtml(clip)).join("")}
                    <div class="h3te-playhead" style="left:${this.playhead * this.pixelsPerSecond}px"></div>
                </div></div>
            </div>`).join("");
        this.syncHorizontalScroll();
    }

    rulerHtml(width) {
        const step = this.pixelsPerSecond >= 160 ? 0.5 : 1;
        const ticks = [];
        for (let time = 0; time <= this.data.duration + 0.0001; time += step) {
            const major = Math.abs(time - Math.round(time)) < 0.001;
            ticks.push(`<span class="h3te-tick ${major ? "major" : "minor"}" style="left:${time * this.pixelsPerSecond}px">${major ? `${Math.round(time)}s` : ""}</span>`);
        }
        return `${ticks.join("")}<div class="h3te-playhead h3te-ruler-head" style="left:${this.playhead * this.pixelsPerSecond}px"></div>`;
    }

    trackIcon(track) {
        if (track.type === TYPES.actorTrack) return "人";
        if (track.type === TYPES.environmentTrack) return "景";
        const kind = track.clips[0]?.kind;
        return kind === "camera" ? "镜" : kind === "lighting" ? "光" : kind === "audio" ? "声" : "系";
    }

    clipHtml(clip) {
        const left = Math.max(0, clip.start) * this.pixelsPerSecond;
        const width = Math.max(12, (clip.end - clip.start) * this.pixelsPerSecond);
        const selected = this.selection?.kind === "clip" && this.selection.node.id === clip.node.id;
        const title = `${KIND_NAMES[clip.kind] || clip.kind} · ${clip.start.toFixed(2)}–${clip.end.toFixed(2)} 秒`;
        const active = this.playhead >= clip.start && this.playhead < clip.end;
        return `<button class="h3te-clip kind-${clip.kind} ${clip.conflict || clip.invalid ? "has-conflict" : ""} ${selected ? "is-selected" : ""} ${active ? "is-active-at-playhead" : ""}" data-node-id="${clip.node.id}" style="left:${left}px;width:${width}px" title="${escapeAttribute(title)}">
            <span class="h3te-resize left" data-resize="left"></span><b>${escapeHtml(KIND_NAMES[clip.kind] || clip.kind)}</b><span>${escapeHtml(clipText(clip.node))}</span>${clip.resultNode ? "<i>缓存</i>" : ""}<span class="h3te-resize right" data-resize="right"></span>
        </button>`;
    }

    markConflicts() {
        for (const track of this.data.tracks) {
            for (const clip of track.clips) {
                clip.conflict = false;
                clip.invalid = clip.start < 0 || clip.end <= clip.start || clip.end > this.data.duration + 0.0001;
            }
            const sorted = [...track.clips].sort((a, b) => a.start - b.start);
            for (let i = 0; i < sorted.length; i++) {
                for (let j = i + 1; j < sorted.length && sorted[j].start < sorted[i].end - 0.0001; j++) {
                    if (sorted[i].kind === sorted[j].kind && sorted[i].kind !== "audio") {
                        sorted[i].conflict = true;
                        sorted[j].conflict = true;
                    }
                }
            }
        }
    }

    syncHorizontalScroll() {
        const rulerScroll = $(".h3te-ruler-scroll", this.root);
        const lanes = [...this.root.querySelectorAll(".h3te-lane-scroll")];
        let syncing = false;
        const synchronize = (source) => {
            if (syncing) return;
            syncing = true;
            const left = source.scrollLeft;
            if (source !== rulerScroll) rulerScroll.scrollLeft = left;
            lanes.forEach((lane) => { if (lane !== source) lane.scrollLeft = left; });
            requestAnimationFrame(() => syncing = false);
        };
        lanes.forEach((lane) => lane.addEventListener("scroll", () => synchronize(lane), { passive: true }));
        rulerScroll.addEventListener("scroll", () => synchronize(rulerScroll), { passive: true });
    }

    renderInspector() {
        const host = $("[data-role=inspector]", this.root);
        const media = $("[data-role=library-media]", this.root);
        if (media) this.renderReferenceAssets(media);
        if (!this.selection) {
            host.innerHTML = `<div class="h3te-empty"><strong>选择要编辑的内容</strong><br>点击时间轴片段，或左侧的人物、环境与风格资源。</div>`;
            return;
        }
        if (this.selection.kind === "video_asset") {
            host.innerHTML = this.referenceAssetForm(this.selection.asset);
            return;
        }
        const nodes = this.selection.nodes || [this.selection.node];
        host.innerHTML = nodes.map((node, index) => this.nodeForm(node, index > 0)).join("") + (this.selection.kind === "clip" ? this.referencePanel(this.selection.node) : "");
    }

    nodeForm(node, secondary) {
        const fields = (FIELD_SETS[nodeType(node)] || []).filter(([name]) =>
            !["use_previous_context", "audio_only_context"].includes(name) || Number(widgetValue(node, "start_time", 0)) > 1e-6);
        const cardTitles = { [TYPES.character]: "人物卡", [TYPES.environment]: "环境卡", [TYPES.visual]: "风格卡" };
        const title = secondary ? cardTitles[nodeType(node)] || node.title || nodeType(node) : this.selection.title || node.title || nodeType(node);
        const saveActor = !secondary && nodeType(node) === TYPES.actor ? `<button data-save-actor-card="${node.id}">保存为人物卡</button>` : "";
        const characterCard = nodeType(node) === TYPES.actor ? findCardForActor(node) : nodeType(node) === TYPES.character ? node : null;
        const referenceCard = characterCard || (nodeType(node) === TYPES.environmentInstance ? findCardForEnvironment(node) : [TYPES.environment, TYPES.visual].includes(nodeType(node)) ? node : null);
        const characterPicker = !secondary && characterCard ? this.characterCardPicker(characterCard) : "";
        const imagePicker = !secondary && referenceCard ? this.cardImagePicker(referenceCard) : "";
        const deleteButton = !secondary && this.selection.kind === "resource" ? `<button class="danger" data-delete-scene-node="${node.id}">删除</button>` : !secondary && this.selection.kind === "clip" ? `<button class="danger" data-delete-clip="${node.id}">删除</button>` : "";
        return `<section class="h3te-form-section ${secondary ? "secondary" : ""}">
            <div class="h3te-form-title"><div><small>${escapeHtml(nodeType(node))} · #${node.id}</small><strong>${escapeHtml(title)}</strong></div><div class="h3te-form-actions">${saveActor}${deleteButton}<button data-locate="${node.id}">定位节点</button></div></div>
            ${characterPicker}
            ${imagePicker}
            ${fields.map(([name, label, hint]) => this.fieldHtml(node, name, label, hint)).join("") || `<div class="h3te-empty">此节点没有可在编辑器中修改的文字属性。</div>`}
        </section>`;
    }

    fieldHtml(node, name, label, hint) {
        const item = widget(node, name);
        if (!item) return "";
        const connected = node.inputs?.some((input) => input.name === name && input.link != null);
        const values = item.options?.values;
        const attrs = `data-field-node="${node.id}" data-field-name="${escapeHtml(name)}" ${connected ? "disabled" : ""}`;
        let control;
        if (Array.isArray(values)) {
            control = `<select ${attrs}>${values.map((value) => `<option ${value === item.value ? "selected" : ""}>${escapeHtml(value)}</option>`).join("")}</select>`;
        } else if (typeof item.value === "boolean") {
            control = `<select ${attrs}><option value="true" ${item.value ? "selected" : ""}>启用</option><option value="false" ${item.value ? "" : "selected"}>关闭</option></select>`;
        } else if (typeof item.value === "number") {
            control = `<input type="number" step="${item.options?.step || 0.05}" value="${item.value}" ${attrs}>`;
        } else if (hint === "textarea") {
            control = `<textarea rows="4" ${attrs}>${escapeHtml(item.value)}</textarea>`;
        } else {
            control = `<input type="text" value="${escapeAttribute(item.value)}" ${attrs}>`;
        }
        return `<label class="h3te-field"><span>${escapeHtml(label)}${connected ? " · 由连线控制" : ""}</span>${control}</label>`;
    }

    referenceVideoUrl(path) {
        const normalized = String(path || "").replaceAll("\\", "/");
        const parts = normalized.split("/").filter(Boolean);
        const filename = parts.pop();
        if (!filename) return "";
        const query = new URLSearchParams({ filename, subfolder: parts.join("/"), type: "input" });
        return api.apiURL(`/view?${query}`);
    }

    referenceImageUrl(reference) {
        if (!reference?.filename) return "";
        const query = new URLSearchParams({ filename: reference.filename, subfolder: reference.subfolder || "", type: reference.type || "input" });
        return api.apiURL(`/view?${query}`);
    }

    referencePanel(node) {
        const config = REFERENCE_SLOTS[nodeType(node)];
        if (!config) return "";
        const reference = linkedNode(node, config.input);
        const split = nodeType(reference) === TYPES.videoPerson ? linkedNode(reference, "motion_reference") : reference;
        const source = split ? linkedNode(split, "video") : null;
        const file = source ? widgetValue(source, "file", "") : "";
        const preview = file ? `<video class="h3te-video-preview" data-timeline-preview controls preload="metadata" src="${escapeAttribute(this.referenceVideoUrl(file))}"></video><small>${escapeHtml(file)}</small>` : reference ? `<div class="h3te-empty">已连接外部参考视频，但其来源不是可直接预览的原生加载视频节点。</div>` : `<div class="h3te-empty">尚未连接${config.label}参考视频。</div>`;
        const assetPicker = this.referenceAssetPicker(node, reference, split, source);
        return `<section class="h3te-form-section secondary h3te-reference-panel">
            <div class="h3te-form-title"><div><small>Ref2VA 语义输入</small><strong>${config.label}参考视频</strong></div>${reference ? `<button data-locate="${reference.id}">定位${nodeType(reference) === TYPES.videoPerson ? "人物解释" : "拆分"}节点</button>` : ""}</div>
            ${preview}
            ${nodeType(reference) === TYPES.videoPerson ? this.fieldHtml(reference, "person_id", "源人物编号") + this.fieldHtml(reference, "person_description", "源人物识别描述", "textarea") : ""}
            ${split ? this.fieldHtml(split, "trim_start", "参考截取开始（秒）") + this.fieldHtml(split, "trim_end", "参考截取结束（秒，0为结尾）") : ""}
            ${assetPicker}
            <label class="h3te-field"><span>${reference ? "替换参考视频" : "上传参考视频"}</span><input type="file" accept="video/*" data-reference-file="${node.id}"></label>
            <button data-attach-reference="${node.id}">${reference ? "上传并替换连接" : "上传并创建参考节点"}</button>
        </section>`;
    }

    referenceAssetPicker(node, reference, split, source) {
        if (!Array.isArray(this.referenceAssets)) {
            return `<div class="h3te-empty">视频资产库尚未载入。请在“参考视频”窗口刷新资产库。</div>`;
        }
        const action = nodeType(node) === TYPES.action;
        const currentAsset = reference?.properties?.minimax_h3_reference_asset
            || split?.properties?.minimax_h3_reference_asset || source?.properties?.minimax_h3_reference_asset || "";
        const currentPerson = reference?.properties?.minimax_h3_reference_person || "";
        const choices = [];
        for (const asset of this.referenceAssets) {
            if (action) {
                for (const person of asset.people || []) {
                    const value = JSON.stringify([asset.id, person.id]);
                    const selected = asset.id === currentAsset && person.id === currentPerson;
                    choices.push(`<option value="${escapeAttribute(value)}" ${selected ? "selected" : ""}>${escapeHtml(asset.display_name)} · ${escapeHtml(person.id)} — ${escapeHtml(shortText(person.description, 42))}</option>`);
                }
            } else {
                const value = JSON.stringify([asset.id, null]);
                choices.push(`<option value="${escapeAttribute(value)}" ${asset.id === currentAsset ? "selected" : ""}>${escapeHtml(asset.display_name)} · ${Number(asset.duration).toFixed(2)} 秒</option>`);
            }
        }
        const empty = action ? "资产库中没有已声明人物的视频。请先在参考视频资产属性中添加源人物。" : "视频资产库为空，请先导入并预处理视频。";
        return `<div class="h3te-reference-asset-picker">
            <label class="h3te-field"><span>直接选择视频资产${action ? "与源人物" : ""}</span><select data-reference-asset-choice="${node.id}" ${choices.length ? "" : "disabled"}><option value="">请选择…</option>${choices.join("")}</select></label>
            <button data-attach-reference-asset="${node.id}" ${choices.length ? "" : "disabled"}>${reference ? "使用资产替换当前参考" : "连接所选视频资产"}</button>
            ${choices.length ? "" : `<small>${empty}</small>`}
        </div>`;
    }

    attachSelectedReferenceAsset(nodeId) {
        const clip = app.graph.getNodeById(Number(nodeId));
        const select = $(`[data-reference-asset-choice="${nodeId}"]`, this.root);
        if (!clip || !select?.value) {
            this.setStatus("请先选择视频资产" + (nodeType(clip) === TYPES.action ? "和源人物" : ""));
            return;
        }
        let choice;
        try {
            choice = JSON.parse(select.value);
        } catch {
            this.setStatus("视频资产选择无效，请刷新资产库后重试");
            return;
        }
        const selectedClip = this.data?.tracks.flatMap((track) => track.clips).find((item) => item.node === clip);
        if (selectedClip) this.selection = { kind: "clip", node: clip, nodes: [clip], title: `${KIND_NAMES[selectedClip.kind] || selectedClip.kind}片段` };
        this.instantiateReferenceAsset(choice[0], true, choice[1]);
    }

    async uploadReferenceVideo(file) {
        const body = new FormData();
        body.append("image", file, file.name);
        body.append("type", "input");
        body.append("subfolder", "minimax_h3/references");
        body.append("overwrite", "false");
        const response = await api.fetchApi("/upload/image", { method: "POST", body });
        if (!response.ok) throw new Error(await response.text() || "参考视频上传失败");
        const uploaded = await response.json();
        return uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name}` : uploaded.name;
    }

    async attachReferenceVideo(nodeId) {
        const clip = app.graph.getNodeById(Number(nodeId));
        const config = REFERENCE_SLOTS[nodeType(clip)];
        const input = $(`[data-reference-file="${nodeId}"]`, this.root);
        const file = input?.files?.[0];
        if (!clip || !config || !file) {
            this.setStatus("请先选择参考视频文件");
            return;
        }
        const button = $(`[data-attach-reference="${nodeId}"]`, this.root);
        button.disabled = true;
        try {
            const path = await this.uploadReferenceVideo(file);
            const created = [];
            this.pendingCreatedNodes = created;
            app.graph.beforeChange?.();
            try {
                const previousReference = linkedNode(clip, config.input);
                const previousSplit = nodeType(previousReference) === TYPES.videoPerson ? linkedNode(previousReference, "motion_reference") : previousReference;
                const previousLoader = previousSplit ? linkedNode(previousSplit, "video") : null;
                const loader = this.createNode(TYPES.loadVideo, [clip.pos[0] - 760, clip.pos[1]], { file: path });
                const reference = this.createNode(TYPES.motionReference, [clip.pos[0] - 460, clip.pos[1]], {
                    trim_start: 0,
                    trim_end: Math.min(15, Math.max(0.05, Number(widgetValue(clip, "end_time", 1)) - Number(widgetValue(clip, "start_time", 0)))),
                });
                if (!connectNodes(loader, loader.outputs?.[0]?.name, reference, "video")) throw new Error("无法连接加载视频与语义拆分节点");
                let semanticNode = reference;
                let semanticOutput = config.output;
                if (nodeType(clip) === TYPES.action) {
                    semanticNode = this.createNode(TYPES.videoPerson, [clip.pos[0] - 220, clip.pos[1]], {
                        person_id: nodeType(previousReference) === TYPES.videoPerson ? widgetValue(previousReference, "person_id", "person_1") : "person_1",
                        person_description: nodeType(previousReference) === TYPES.videoPerson ? widgetValue(previousReference, "person_description", "the primary performer in the reference video") : "the primary performer in the reference video",
                    });
                    semanticOutput = 0;
                    if (!connectNodes(reference, reference.outputs?.[0]?.name, semanticNode, "motion_reference")) throw new Error("无法连接参考视频人物解释");
                }
                const inputIndex = slotIndex(clip, config.input);
                if (inputIndex < 0 || !semanticNode.outputs?.[semanticOutput]) throw new Error("片段不支持所选参考视频语义");
                semanticNode.connect(semanticOutput, clip, inputIndex);
                loader.properties ||= {};
                loader.properties.minimax_h3_reference_upload = true;
                reference.properties ||= {};
                reference.properties.minimax_h3_reference_upload = true;
                if (semanticNode !== reference) {
                    semanticNode.properties ||= {};
                    semanticNode.properties.minimax_h3_reference_upload = true;
                }
                if (isManagedMediaNode(previousReference) && outputTargets(previousReference).length === 0) app.graph.remove(previousReference);
                if (previousSplit !== previousReference && isManagedMediaNode(previousSplit) && outputTargets(previousSplit).length === 0) app.graph.remove(previousSplit);
                if (isManagedMediaNode(previousLoader) && outputTargets(previousLoader).length === 0) app.graph.remove(previousLoader);
                this.layoutProjectNodes(false);
                app.graph.afterChange?.();
                app.graph.setDirtyCanvas(true, true);
                this.markWorkflowChanged();
                this.data = collectTimeline(this.currentTimeline());
                this.renderTracks();
                this.renderInspector();
                this.setStatus(`${config.label}参考视频已上传、连接并可预览`);
            } catch (error) {
                for (const node of created.reverse()) app.graph.remove(node);
                app.graph.afterChange?.();
                throw error;
            } finally {
                this.pendingCreatedNodes = null;
            }
        } catch (error) {
            this.setStatus(`参考视频处理失败：${error?.message || error}`);
            button.disabled = false;
        }
    }

    onClick(event) {
        const dockTab = event.target.closest("[data-dock-tab]");
        if (dockTab) return this.activateDockPanel(dockTab.dataset.dockTab);
        const createResource = event.target.closest("[data-create-resource]");
        if (createResource) return this.openResourceModal(createResource.dataset.createResource);
        const editResource = event.target.closest("[data-edit-resource]");
        if (editResource) {
            const [kind, id] = editResource.dataset.editResource.split(":");
            return this.openResourceModal(kind, id);
        }
        const deleteResource = event.target.closest("[data-delete-resource]");
        if (deleteResource) {
            const [kind, id] = deleteResource.dataset.deleteResource.split(":");
            return this.deleteLibraryResource(kind, id);
        }
        const instantiate = event.target.closest("[data-instantiate-resource]");
        if (instantiate) {
            const [kind, id] = instantiate.dataset.instantiateResource.split(":");
            return this.instantiateResource(kind, id);
        }
        const instantiateReference = event.target.closest("[data-instantiate-reference-asset]");
        if (instantiateReference) return this.instantiateReferenceAsset(instantiateReference.dataset.instantiateReferenceAsset, false);
        const useReference = event.target.closest("[data-use-reference-asset]");
        if (useReference) return this.instantiateReferenceAsset(useReference.dataset.useReferenceAsset, true);
        const useReferencePerson = event.target.closest("[data-use-reference-person]");
        if (useReferencePerson) {
            const [assetId, personId] = useReferencePerson.dataset.useReferencePerson.split(":");
            return this.instantiateReferenceAsset(assetId, true, personId);
        }
        const editReference = event.target.closest("[data-edit-reference-asset]");
        if (editReference) return this.editReferenceAsset(editReference.dataset.editReferenceAsset);
        const addReferencePerson = event.target.closest("[data-add-reference-person]");
        if (addReferencePerson) return this.addReferencePerson();
        const removeReferencePerson = event.target.closest("[data-remove-reference-person]");
        if (removeReferencePerson) return this.removeReferencePerson(Number(removeReferencePerson.dataset.removeReferencePerson));
        const saveReferenceAsset = event.target.closest("[data-save-reference-asset]");
        if (saveReferenceAsset) return this.saveReferenceAsset(saveReferenceAsset.dataset.saveReferenceAsset);
        const deleteReference = event.target.closest("[data-delete-reference-asset]");
        if (deleteReference) return this.deleteReferenceAsset(deleteReference.dataset.deleteReferenceAsset);
        const action = event.target.closest("[data-action]")?.dataset.action;
        if (action === "confirm-delete") return this.closeConfirm(true);
        if (action === "cancel-delete") return this.closeConfirm(false);
        if (action === "close-resource-modal") {
            this.resourceModal = null;
            return this.renderResourceModal();
        }
        if (action === "close-track-modal") {
            this.trackModal = null;
            return this.renderTrackModal();
        }
        if (action === "close-reference-asset-modal") {
            this.referenceAssetModal = false;
            return this.renderReferenceAssetModal();
        }
        if (action === "import-reference-asset") {
            this.referenceAssetModal = true;
            return this.renderReferenceAssetModal();
        }
        if (action === "reload-reference-assets") return this.loadReferenceAssets();
        if (action === "reload-library") return this.loadResourceLibrary();
        if (action === "close") return this.close();
        if (action === "refresh") {
            this.layoutProjectNodes(true);
            return this.render();
        }
        if (action === "save") return this.saveWorkflow(true);
        if (action === "export-project") return this.exportProject();
        if (action === "package-project") return this.packageProject();
        if (action === "import-project") return $("[data-role=project-file]", this.root).click();
        if (action === "bootstrap") return this.bootstrapProject();
        if (action === "cleanup") return this.cleanupUnusedNodes();
        if (action === "create-track") return this.openTrackModal();
        if (action === "fit" && this.data) {
            const available = Math.max(480, $(".h3te-stage", this.root).clientWidth - 190);
            this.pixelsPerSecond = Math.max(60, Math.min(260, available / this.data.duration));
            $("[data-role=zoom]", this.root).value = this.pixelsPerSecond;
            return this.renderTracks();
        }
        if (action === "play") return this.togglePlay();

        const addClip = event.target.closest("[data-add-clip]");
        if (addClip) return this.openTrackModal(Number(addClip.dataset.addClip));
        const attachReference = event.target.closest("[data-attach-reference]");
        if (attachReference) return this.attachReferenceVideo(attachReference.dataset.attachReference);
        const attachReferenceAsset = event.target.closest("[data-attach-reference-asset]");
        if (attachReferenceAsset) return this.attachSelectedReferenceAsset(attachReferenceAsset.dataset.attachReferenceAsset);
        const saveActorCard = event.target.closest("[data-save-actor-card]");
        if (saveActorCard) return this.saveActorAsResource(app.graph.getNodeById(Number(saveActorCard.dataset.saveActorCard)));
        const applyCharacterCard = event.target.closest("[data-apply-character-card]");
        if (applyCharacterCard) return this.applyCharacterCard(applyCharacterCard.dataset.applyCharacterCard);
        const selectAnimaCharacter = event.target.closest("[data-select-anima-character]");
        if (selectAnimaCharacter) return this.selectAnimaCharacter(selectAnimaCharacter.dataset.selectAnimaCharacter);
        const attachCardImage = event.target.closest("[data-attach-card-image]");
        if (attachCardImage) return this.attachCardImage(attachCardImage.dataset.attachCardImage);
        const deleteSceneNode = event.target.closest("[data-delete-scene-node]");
        if (deleteSceneNode) return this.deleteSceneNode(deleteSceneNode.dataset.deleteSceneNode);
        const deleteTrack = event.target.closest("[data-delete-track]");
        if (deleteTrack) return this.deleteTrack(deleteTrack.dataset.deleteTrack);
        const deleteClip = event.target.closest("[data-delete-clip]");
        if (deleteClip) return this.deleteClip(deleteClip.dataset.deleteClip);

        const resourceButton = event.target.closest("[data-resource]");
        if (resourceButton) {
            const resource = this.resources[Number(resourceButton.dataset.resource)];
            const nodes = resource.card && resource.card !== resource.node ? [resource.node, resource.card] : [resource.node];
            this.selection = { kind: "resource", node: resource.node, nodes, title: resource.title };
            return this.renderInspector();
        }
        const trackButton = event.target.closest("[data-track]");
        if (trackButton) {
            const track = this.data.tracks[Number(trackButton.dataset.track)];
            if (track.owner) {
                const card = track.type === TYPES.actorTrack ? findCardForActor(track.owner) : findCardForEnvironment(track.owner);
                this.selection = { kind: "resource", node: track.owner, nodes: card ? [track.owner, card] : [track.owner], title: track.label };
                this.renderInspector();
            }
            return;
        }
        const locate = event.target.closest("[data-locate]");
        if (locate) return this.locateNode(Number(locate.dataset.locate));

    }

    onFieldChange(event) {
        const assetField = event.target.closest("[data-reference-asset-field]");
        if (assetField && this.selection?.kind === "video_asset") {
            this.selection.asset[assetField.dataset.referenceAssetField] = assetField.value;
            return;
        }
        const personField = event.target.closest("[data-reference-person-field]");
        if (personField && this.selection?.kind === "video_asset") {
            const person = this.selection.asset.people?.[Number(personField.dataset.referencePersonIndex)];
            if (person) person[personField.dataset.referencePersonField] = personField.value;
            return;
        }
        const field = event.target.closest("[data-field-node]");
        if (!field) return;
        const node = app.graph.getNodeById(Number(field.dataset.fieldNode));
        const current = widget(node, field.dataset.fieldName)?.value;
        const value = typeof current === "number" ? Number(field.value) : typeof current === "boolean" ? field.value === "true" : field.value;
        if (setWidgetValue(node, field.dataset.fieldName, value)) {
            this.data = collectTimeline(this.currentTimeline());
            if (["start_time", "end_time", "action_type", "name"].includes(field.dataset.fieldName)) {
                this.renderLibrary();
                this.renderTracks();
                if (field.dataset.fieldName === "start_time") this.renderInspector();
            }
            this.setStatus("已同步到工作流");
        }
    }

    onPointerDown(event) {
        const resizer = event.target.closest("[data-dock-resize]");
        if (resizer) {
            event.preventDefault();
            const [splitId, rawIndex] = resizer.dataset.dockResize.split(":");
            const layout = this.findDockNode(splitId);
            const split = resizer.parentElement;
            const cells = [...split.children].filter((item) => item.classList.contains("h3te-dock-cell"));
            const index = Number(rawIndex);
            if (!layout || !cells[index] || !cells[index + 1]) return;
            this.dockResize = {
                layout, split, resizer, cells, index,
                start: layout.direction === "row" ? event.clientX : event.clientY,
                first: layout.sizes[index], second: layout.sizes[index + 1],
            };
            resizer.classList.add("is-dragging");
            return;
        }
        const clipElement = event.target.closest(".h3te-clip");
        if (clipElement) {
            event.preventDefault();
            const node = app.graph.getNodeById(Number(clipElement.dataset.nodeId));
            const clip = this.data.tracks.flatMap((track) => track.clips).find((item) => item.node === node);
            if (!clip) return;
            this.selection = { kind: "clip", node, nodes: [node], title: `${KIND_NAMES[clip.kind] || clip.kind}片段${clip.resultNode ? " · 已缓存" : ""}` };
            this.renderInspector();
            this.drag = {
                clip,
                mode: event.target.dataset.resize || "move",
                startX: event.clientX,
                start: clip.start,
                end: clip.end,
                element: clipElement,
            };
            clipElement.setPointerCapture?.(event.pointerId);
            clipElement.classList.add("is-dragging");
            return;
        }
        const surface = event.target.closest(".h3te-ruler,.h3te-lane");
        if (surface && this.data) {
            event.preventDefault();
            if (this.playing) this.togglePlay();
            this.scrub = { surface, pointerId: event.pointerId };
            surface.setPointerCapture?.(event.pointerId);
            this.seekFromPointer(event, surface);
        }
    }

    onPointerMove(event) {
        if (this.dockResize) {
            const state = this.dockResize;
            const pixels = state.layout.direction === "row" ? state.split.clientWidth : state.split.clientHeight;
            const current = state.layout.direction === "row" ? event.clientX : event.clientY;
            const total = state.layout.sizes.reduce((sum, value) => sum + value, 0);
            const pair = state.first + state.second;
            const minimum = Math.min(pair / 2, Math.max(pair * 0.12, total * 0.04));
            const first = Math.max(minimum, Math.min(pair - minimum, state.first + (current - state.start) * total / Math.max(1, pixels)));
            state.layout.sizes[state.index] = first;
            state.layout.sizes[state.index + 1] = pair - first;
            state.cells[state.index].style.flexBasis = `${100 * first / total}%`;
            state.cells[state.index + 1].style.flexBasis = `${100 * (pair - first) / total}%`;
            return;
        }
        if (this.scrub) {
            this.seekFromPointer(event, this.scrub.surface);
            return;
        }
        if (!this.drag || !this.data) return;
        const delta = (event.clientX - this.drag.startX) / this.pixelsPerSecond;
        const duration = this.drag.end - this.drag.start;
        let start = this.drag.start;
        let end = this.drag.end;
        if (this.drag.mode === "left") start = Math.min(end - 0.05, this.drag.start + delta);
        else if (this.drag.mode === "right") end = Math.max(start + 0.05, this.drag.end + delta);
        else {
            start = this.drag.start + delta;
            end = start + duration;
            if (start < 0) { start = 0; end = duration; }
            if (end > this.data.duration) { end = this.data.duration; start = end - duration; }
        }
        start = this.snapTime(Math.max(0, start));
        end = this.snapTime(Math.min(this.data.duration, end));
        if (end <= start) end = Math.min(this.data.duration, start + 0.05);
        this.drag.currentStart = start;
        this.drag.currentEnd = end;
        this.drag.element.style.left = `${start * this.pixelsPerSecond}px`;
        this.drag.element.style.width = `${Math.max(12, (end - start) * this.pixelsPerSecond)}px`;
        this.drag.element.title = `${start.toFixed(2)}–${end.toFixed(2)} 秒`;
        this.setStatus(`正在调整：${start.toFixed(2)}–${end.toFixed(2)} 秒`);
    }

    onPointerUp(event) {
        if (this.dockResize) {
            this.dockResize.resizer.classList.remove("is-dragging");
            this.dockResize = null;
            this.saveDockLayout();
            this.renderTracks();
            this.setStatus("窗口边界已调整并保存");
            return;
        }
        if (this.scrub) {
            this.scrub.surface.releasePointerCapture?.(this.scrub.pointerId ?? event?.pointerId);
            this.scrub = null;
            this.setStatus(`播放头：${this.formatTimecode(this.playhead)}`);
            return;
        }
        if (!this.drag) return;
        const start = this.drag.currentStart ?? this.drag.start;
        const end = this.drag.currentEnd ?? this.drag.end;
        setWidgetValue(this.drag.clip.node, "start_time", start);
        setWidgetValue(this.drag.clip.node, "end_time", end);
        this.drag.element.classList.remove("is-dragging");
        this.drag = null;
        this.data = collectTimeline(this.currentTimeline());
        this.renderTracks();
        this.renderInspector();
        this.setStatus("片段时间已同步到工作流");
    }

    snapTime(value) {
        return this.snap ? Math.round(value / this.snap) * this.snap : value;
    }

    seekFromPointer(event, surface) {
        const rect = surface.getBoundingClientRect();
        this.setPlayhead((event.clientX - rect.left) / this.pixelsPerSecond);
    }

    setPlayhead(value, { scrollIntoView = false } = {}) {
        if (!this.data) return;
        this.playhead = Math.max(0, Math.min(this.data.duration, Number(value) || 0));
        this.updatePlayheadVisuals();
        this.syncTimelinePreview();
        if (scrollIntoView) this.scrollPlayheadIntoView();
    }

    updatePlayheadVisuals() {
        const left = `${this.playhead * this.pixelsPerSecond}px`;
        this.root.querySelectorAll(".h3te-playhead").forEach((line) => line.style.left = left);
        for (const element of this.root.querySelectorAll(".h3te-clip")) {
            const clip = this.data?.tracks.flatMap((track) => track.clips).find((item) => item.node.id === Number(element.dataset.nodeId));
            element.classList.toggle("is-active-at-playhead", Boolean(clip && this.playhead >= clip.start && this.playhead < clip.end));
        }
        this.updateTime();
    }

    scrollPlayheadIntoView() {
        const ruler = $(".h3te-ruler-scroll", this.root);
        if (!ruler) return;
        const x = this.playhead * this.pixelsPerSecond;
        const margin = Math.min(120, ruler.clientWidth * 0.2);
        if (x < ruler.scrollLeft + margin) ruler.scrollLeft = Math.max(0, x - margin);
        else if (x > ruler.scrollLeft + ruler.clientWidth - margin) ruler.scrollLeft = x - ruler.clientWidth + margin;
        ruler.dispatchEvent(new Event("scroll"));
    }

    syncTimelinePreview() {
        const video = $("[data-timeline-preview]", this.root);
        if (!video || this.selection?.kind !== "clip") return;
        const clip = this.data?.tracks.flatMap((track) => track.clips).find((item) => item.node === this.selection.node);
        if (!clip || this.playhead < clip.start || this.playhead > clip.end || !Number.isFinite(video.duration)) return;
        const config = REFERENCE_SLOTS[nodeType(clip.node)];
        const reference = config ? linkedNode(clip.node, config.input) : null;
        const split = nodeType(reference) === TYPES.videoPerson ? linkedNode(reference, "motion_reference") : reference;
        const trimStart = Math.max(0, Number(widgetValue(split, "trim_start", 0)) || 0);
        const trimEndValue = Number(widgetValue(split, "trim_end", 0)) || 0;
        const trimEnd = trimEndValue > trimStart ? Math.min(video.duration, trimEndValue) : video.duration;
        const progress = (this.playhead - clip.start) / Math.max(0.001, clip.end - clip.start);
        const target = trimStart + progress * Math.max(0, trimEnd - trimStart);
        if (Math.abs(video.currentTime - target) > 1 / 48) video.currentTime = target;
    }

    formatTimecode(value) {
        const totalFrames = Math.max(0, Math.round(value * 24));
        const frames = totalFrames % 24;
        const totalSeconds = Math.floor(totalFrames / 24);
        const seconds = totalSeconds % 60;
        const minutes = Math.floor(totalSeconds / 60) % 60;
        const hours = Math.floor(totalSeconds / 3600);
        return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}:${String(frames).padStart(2, "0")}`;
    }

    togglePlay() {
        if (!this.data) return;
        this.playing = !this.playing;
        $("[data-action=play]", this.root).textContent = this.playing ? "❚❚" : "▶";
        if (!this.playing) return;
        if (this.playhead >= this.data.duration) this.setPlayhead(0);
        let previous = performance.now();
        const frame = (now) => {
            if (!this.playing) return;
            const next = this.playhead + (now - previous) / 1000;
            previous = now;
            if (next >= this.data.duration) {
                this.playing = false;
                $("[data-action=play]", this.root).textContent = "▶";
            }
            this.setPlayhead(Math.min(next, this.data.duration), { scrollIntoView: true });
            if (this.playing) requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
    }

    updateTime() {
        $("[data-role=time]", this.root).textContent = this.formatTimecode(this.playhead);
    }

    locateNode(id) {
        const node = app.graph?.getNodeById(id);
        if (!node) return;
        this.close();
        app.canvas.selectNode(node);
        app.canvas.centerOnNode(node);
        app.graph.setDirtyCanvas(true, false);
    }

    setStatus(text) {
        $("[data-role=status]", this.root).textContent = text;
    }

    setSaveStatus(text) {
        const status = $("[data-role=save-status]", this.root);
        if (status) status.textContent = text;
    }
}

let editor;
function openEditor(timeline) {
    editor ||= new H3TimelineEditor();
    editor.open(timeline);
}

app.registerExtension({
    name: "MiniMaxH3.TimelineEditor",
    commands: [{
        id: "MiniMaxH3.openTimelineEditor",
        label: "打开 MiniMax H3 导演时间轴",
        menubarLabel: "导演时间轴编辑器",
        function: () => openEditor(),
    }],
    menuCommands: [{ path: ["MiniMax H3"], commands: ["MiniMaxH3.openTimelineEditor"] }],
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === TYPES.action) {
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                migrateActionContextWidget(this);
            };
            return;
        }
        if (nodeData.name !== TYPES.timeline) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const button = this.addWidget("button", "打开导演时间轴", null, () => openEditor(this));
            button.serialize = false;
        };
    },
});
