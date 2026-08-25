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
    motionReference: "MiniMaxH3MotionReference",
    loadVideo: "LoadVideo",
};

const FIELD_SETS = {
    [TYPES.action]: [
        ["action_type", "动作种类"], ["start_time", "开始时间（秒）"], ["end_time", "结束时间（秒）"],
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
};

const KIND_NAMES = {
    body: "肢体", expression: "表情", gaze: "视线", speech: "对话",
    camera: "镜头", lighting: "灯光", audio: "音频", environment: "环境",
};

const CLIP_TYPES = new Set([TYPES.action, TYPES.camera, TYPES.lighting, TYPES.audio, TYPES.environmentAction]);
const RESOURCE_NAMES = { characters: "人物", environments: "环境", styles: "风格" };
const TRACK_KIND_NAMES = { body: "人物·肢体", expression: "人物·表情", gaze: "人物·视线", environment: "环境", camera: "镜头", lighting: "灯光", audio: "音频" };
const REFERENCE_SLOTS = {
    [TYPES.action]: { input: "motion_reference", output: 0, label: "人物表演" },
    [TYPES.camera]: { input: "camera_reference", output: 1, label: "镜头" },
    [TYPES.lighting]: { input: "lighting_reference", output: 2, label: "灯光" },
    [TYPES.audio]: { input: "audio_reference", output: 3, label: "音频" },
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
        tracks,
        characterGroup: linkedNode(timeline, "character_group"),
        style: linkedNode(timeline, "style_card"),
        environment: linkedNode(timeline, "environment"),
    };
}

function timelineNodes() {
    return (app.graph?._nodes || []).filter((node) => nodeType(node) === TYPES.timeline);
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
        this.drag = null;
        this.library = null;
        this.referenceAssets = null;
        this.libraryTab = "scene";
        this.resourceFilter = "";
        this.resourceModal = null;
        this.referenceAssetModal = false;
        this.trackModal = null;
        this.createdOffset = 0;
        this.autoSave = localStorage.getItem("minimax_h3_timeline_autosave") !== "false";
        this.saveTimer = null;
        this.changeVersion = 0;
        this.root = document.createElement("div");
        this.root.className = "h3te-root";
        this.root.innerHTML = this.shell();
        document.body.appendChild(this.root);
        this.bind();
    }

    shell() {
        return `
            <header class="h3te-header">
                <div class="h3te-brand"><span class="h3te-mark">H3</span><div><strong>导演时间轴</strong><small>MiniMax H3 Prompt Builder</small></div></div>
                <label class="h3te-timeline-select">总时间轴 <select data-role="timeline"></select></label>
                <div class="h3te-transport">
                    <button data-action="play" title="播放时间指针">▶</button>
                    <span data-role="time">00:00.00</span>
                </div>
                <div class="h3te-tools">
                    <label>时长 <input class="h3te-duration" data-role="duration" type="number" min="0.21" max="60" step="0.05"></label>
                    <label>吸附 <select data-role="snap"><option value="0">关闭</option><option value="0.01">0.01秒</option><option value="0.05" selected>0.05秒</option><option value="0.1">0.1秒</option><option value="0.25">0.25秒</option></select></label>
                    <label>缩放 <input data-role="zoom" type="range" min="60" max="260" value="115"></label>
                    <button data-action="fit">适应宽度</button><button data-action="bootstrap">补全工程</button><button data-action="layout">自动排布</button><button data-action="create-track">＋轨道</button>
                    <label class="h3te-autosave"><input data-role="autosave" type="checkbox" ${this.autoSave ? "checked" : ""}>自动保存</label><button data-action="save">保存</button><button data-action="refresh">刷新</button><button class="h3te-close" data-action="close">关闭</button>
                </div>
            </header>
            <main class="h3te-main">
                <aside class="h3te-library">
                    <div class="h3te-panel-title">资源</div>
                    <div class="h3te-library-tabs"><button class="is-active" data-library-tab="scene">当前工程</button><button data-library-tab="resources">卡片</button><button data-library-tab="media">参考视频</button></div>
                    <div data-role="library"></div>
                </aside>
                <section class="h3te-stage">
                    <div class="h3te-ruler-wrap"><div class="h3te-track-label h3te-ruler-label">轨道</div><div class="h3te-ruler-scroll"><div class="h3te-ruler" data-role="ruler"></div></div></div>
                    <div class="h3te-tracks" data-role="tracks"></div>
                </section>
                <aside class="h3te-inspector"><div class="h3te-panel-title">属性</div><div data-role="inspector" class="h3te-inspector-body"></div></aside>
            </main>
            <footer class="h3te-footer"><span data-role="status">修改会直接同步到 ComfyUI 工作流</span><span data-role="save-status">${this.autoSave ? "自动保存已开启" : "自动保存已关闭"}</span><span>拖动片段调整时间 · 拖动两端改变长度</span></footer>
            <div class="h3te-modal-layer" data-role="resource-modal"></div>
            <div class="h3te-modal-layer" data-role="reference-asset-modal"></div>
            <div class="h3te-modal-layer" data-role="track-modal"></div>`;
    }

    bind() {
        this.root.addEventListener("click", (event) => this.onClick(event));
        $("[data-role=timeline]", this.root).addEventListener("change", (event) => {
            this.timelineId = Number(event.target.value);
            this.selection = null;
            this.render();
        });
        $("[data-role=zoom]", this.root).addEventListener("input", (event) => {
            this.pixelsPerSecond = Number(event.target.value);
            this.renderTracks();
        });
        $("[data-role=snap]", this.root).addEventListener("change", (event) => this.snap = Number(event.target.value));
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
        window.addEventListener("pointermove", (event) => this.onPointerMove(event));
        window.addEventListener("pointerup", () => this.onPointerUp());
        window.addEventListener("keydown", (event) => {
            if (event.key !== "Escape" || !this.root.classList.contains("is-open")) return;
            if (this.resourceModal) {
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
        this.root.classList.add("is-open");
        document.body.classList.add("h3te-open");
        this.render();
        this.loadResourceLibrary();
        this.loadReferenceAssets();
    }

    close() {
        this.playing = false;
        this.resourceModal = null;
        this.referenceAssetModal = false;
        this.trackModal = null;
        this.renderResourceModal();
        this.renderReferenceAssetModal();
        this.renderTrackModal();
        this.root.classList.remove("is-open");
        document.body.classList.remove("h3te-open");
    }

    currentTimeline() {
        return app.graph?.getNodeById(this.timelineId);
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
        const host = $("[data-role=library]", this.root);
        this.root.querySelectorAll("[data-library-tab]").forEach((button) => button.classList.toggle("is-active", button.dataset.libraryTab === this.libraryTab));
        if (this.libraryTab === "media") {
            this.renderReferenceAssets(host);
            return;
        }
        if (this.libraryTab === "resources") {
            this.renderResourceLibrary(host);
            return;
        }
        if (!this.data) {
            host.innerHTML = `<div class="h3te-empty">工作流中没有 MiniMax H3 总时间轴节点。</div>`;
            return;
        }
        const items = this.sceneItems();
        host.innerHTML = items.length ? items.map((item, index) => `
            <button class="h3te-resource ${item.color}" data-resource="${index}"><span>${escapeHtml(item.type)}</span><strong>${escapeHtml(item.title)}</strong></button>`).join("") : `<div class="h3te-empty">总时间轴尚未连接场景资源。</div>`;
        this.resources = items;
    }

    async loadResourceLibrary() {
        try {
            const response = await api.fetchApi("/minimax-h3/resources");
            const body = await response.json();
            if (!response.ok || !body.success) throw new Error(body.error || "读取资源库失败");
            this.library = body.library;
            this.renderLibrary();
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
        if (this.libraryTab === "media") this.renderLibrary();
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
        host.innerHTML = `${toolbar}<div class="h3te-resource-count">${this.referenceAssets.length} 个已预处理资产 · 统一为 24 FPS</div>${this.referenceAssets.length ? this.referenceAssets.map((asset) => `
            <article class="h3te-library-card media">
                <video class="h3te-video-preview" controls preload="metadata" src="${escapeAttribute(this.referenceVideoUrl(this.referenceAssetPath(asset)))}"></video>
                <div><span>参考视频资产</span><strong>${escapeHtml(asset.display_name)}</strong><small>${Number(asset.duration).toFixed(2)} 秒 · ${asset.width}×${asset.height} · ${asset.preprocess?.fps || 24} FPS</small></div>
                <div class="h3te-library-actions">${canAttach ? `<button data-use-reference-asset="${asset.id}">用于当前片段</button>` : ""}<button data-instantiate-reference-asset="${asset.id}">创建四路输出</button><button class="danger" data-delete-reference-asset="${asset.id}">移出资产库</button></div>
            </article>`).join("") : `<div class="h3te-empty">尚未导入参考视频。导入时会完成截取、24 FPS 重采样和音频标准化封装。</div>`}`;
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
                <small>导入时会生成独立 MP4 资产，按 24 FPS 预处理并保留原宽高比；单个资产最长 15 秒。</small>
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

    instantiateReferenceAsset(assetId, attachToSelection = false) {
        const asset = Array.isArray(this.referenceAssets) ? this.referenceAssets.find((item) => item.id === assetId) : null;
        const clip = attachToSelection && this.selection?.kind === "clip" ? this.selection.node : null;
        const config = clip ? REFERENCE_SLOTS[nodeType(clip)] : null;
        if (!asset || (attachToSelection && !config)) return;
        const timeline = this.currentTimeline();
        const base = clip?.pos || timeline?.pos || app.canvas.graph_mouse || [700, 300];
        const standaloneOffset = clip ? 0 : this.createdOffset++ * 170;
        const anchor = [base[0], base[1] + standaloneOffset];
        const created = [];
        this.pendingCreatedNodes = created;
        app.graph.beforeChange?.();
        try {
            const loader = this.createNode(TYPES.loadVideo, [anchor[0] - 620, anchor[1]], { file: this.referenceAssetPath(asset) });
            const reference = this.createNode(TYPES.motionReference, [anchor[0] - 310, anchor[1]], { trim_start: 0, trim_end: 0 });
            if (!connectNodes(loader, loader.outputs?.[0]?.name, reference, "video")) throw new Error("无法连接参考视频资产");
            if (clip) {
                const input = slotIndex(clip, config.input);
                if (input < 0 || !reference.outputs?.[config.output]) throw new Error("当前片段不支持该参考语义");
                reference.connect(config.output, clip, input);
            }
            loader.properties ||= {};
            reference.properties ||= {};
            loader.properties.minimax_h3_reference_asset = asset.id;
            reference.properties.minimax_h3_reference_asset = asset.id;
            if (timeline) this.layoutProjectNodes(false);
            app.graph.afterChange?.();
            this.markWorkflowChanged();
            this.render();
            this.setStatus(clip ? `${asset.display_name} 已连接到当前片段` : `${asset.display_name} 已创建四路语义输出节点`);
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
        if (!asset || !confirm(`将“${asset.display_name}”移出资产库？已创建的工作流节点和媒体文件会保留。`)) return;
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
        if (!resource || !confirm(`删除资源“${resource.display_name}”？已实例化的工作流节点不会被删除。`)) return;
        try {
            const next = structuredClone(this.library);
            next[kind] = next[kind].filter((item) => item.id !== id);
            await this.saveResourceLibrary(next);
            this.setStatus("资源已从 JSON 资源库删除，现有工作流节点保持不变");
        } catch (error) {
            this.setStatus(`删除失败：${error?.message || error}`);
        }
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
                const actor = this.createNode(TYPES.actor, [tx - 600, ty - 160]);
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
        const placed = new Set([timeline.id]);
        let changed = false;
        const place = (node, x, y) => {
            if (!node || placed.has(node.id)) return;
            placed.add(node.id);
            if (node.pos[0] !== x || node.pos[1] !== y) {
                node.pos = [x, y];
                changed = true;
            }
        };
        if (record) app.graph.beforeChange?.();
        place(data.style, tx - 300, ty - 500);
        place(linkedNode(data.style, "reference_image"), tx - 600, ty - 500);
        const actors = data.characterGroup ? numberedInputs(data.characterGroup, ["actors", "actor"]) : [];
        const actorTop = ty - 220;
        actors.forEach((actor, index) => {
            const y = actorTop + index * 190;
            const card = findCardForActor(actor);
            place(card, tx - 900, y);
            place(linkedNode(card, "reference_image"), tx - 1200, y);
            place(actor, tx - 600, y);
        });
        place(data.characterGroup, tx - 300, actorTop + Math.max(0, actors.length - 1) * 95);
        const environmentY = actorTop + Math.max(1, actors.length) * 190 + 70;
        const environmentCard = findCardForEnvironment(data.environment);
        place(environmentCard, tx - 600, environmentY);
        place(linkedNode(environmentCard, "reference_image"), tx - 900, environmentY);
        place(data.environment, tx - 300, environmentY);
        let trackY = environmentY + 260;
        const trackList = linkedNode(timeline, "tracks");
        data.tracks.forEach((track) => {
            const laneHeight = Math.max(210, track.clips.length * 150);
            place(track.node, tx - 600, trackY);
            track.clips.forEach((clip, index) => {
                const y = trackY + index * 150;
                place(clip.node, tx - 1050, y);
                place(clip.resultNode, tx - 800, y);
                const config = REFERENCE_SLOTS[nodeType(clip.node)];
                const reference = config ? linkedNode(clip.node, config.input) : null;
                place(reference, tx - 1350, y);
                place(linkedNode(reference, "video"), tx - 1650, y);
            });
            trackY += laneHeight;
        });
        place(trackList, tx - 300, environmentY + 260);
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
        const path = resource.reference_image.subfolder ? `${resource.reference_image.subfolder}/${resource.reference_image.filename}` : resource.reference_image.filename;
        setWidgetValue(loader, "image", path);
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
                const actor = this.createNode(TYPES.actor, [x + 330, y], resource.instance_defaults, resource, kind);
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
            values.content = kind === "body" ? "performs the planned action naturally" : kind === "expression" ? "changes expression naturally" : "shifts gaze naturally";
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
                <div class="h3te-track-head"><button class="h3te-track-label" data-track="${index}"><span>${this.trackIcon(track)}</span><strong>${escapeHtml(track.label)}</strong><small>${track.clips.length} 个片段</small></button><button class="h3te-add-clip" data-add-clip="${index}" title="向此轨道添加片段">＋</button></div>
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
        return `<button class="h3te-clip kind-${clip.kind} ${clip.conflict || clip.invalid ? "has-conflict" : ""} ${selected ? "is-selected" : ""}" data-node-id="${clip.node.id}" style="left:${left}px;width:${width}px" title="${escapeAttribute(title)}">
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
        if (!this.selection) {
            host.innerHTML = `<div class="h3te-empty"><strong>选择要编辑的内容</strong><br>点击时间轴片段，或左侧的人物、环境与风格资源。</div>`;
            return;
        }
        const nodes = this.selection.nodes || [this.selection.node];
        host.innerHTML = nodes.map((node, index) => this.nodeForm(node, index > 0)).join("") + (this.selection.kind === "clip" ? this.referencePanel(this.selection.node) : "");
    }

    nodeForm(node, secondary) {
        const fields = FIELD_SETS[nodeType(node)] || [];
        const cardTitles = { [TYPES.character]: "人物卡", [TYPES.environment]: "环境卡", [TYPES.visual]: "风格卡" };
        const title = secondary ? cardTitles[nodeType(node)] || node.title || nodeType(node) : this.selection.title || node.title || nodeType(node);
        const saveActor = !secondary && nodeType(node) === TYPES.actor ? `<button data-save-actor-card="${node.id}">保存为人物卡</button>` : "";
        return `<section class="h3te-form-section ${secondary ? "secondary" : ""}">
            <div class="h3te-form-title"><div><small>${escapeHtml(nodeType(node))} · #${node.id}</small><strong>${escapeHtml(title)}</strong></div><div class="h3te-form-actions">${saveActor}<button data-locate="${node.id}">定位节点</button></div></div>
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

    referencePanel(node) {
        const config = REFERENCE_SLOTS[nodeType(node)];
        if (!config) return "";
        const reference = linkedNode(node, config.input);
        const source = reference ? linkedNode(reference, "video") : null;
        const file = source ? widgetValue(source, "file", "") : "";
        const preview = file ? `<video class="h3te-video-preview" controls preload="metadata" src="${escapeAttribute(this.referenceVideoUrl(file))}"></video><small>${escapeHtml(file)}</small>` : reference ? `<div class="h3te-empty">已连接外部参考视频，但其来源不是可直接预览的原生加载视频节点。</div>` : `<div class="h3te-empty">尚未连接${config.label}参考视频。</div>`;
        return `<section class="h3te-form-section secondary h3te-reference-panel">
            <div class="h3te-form-title"><div><small>Ref2VA 语义输入</small><strong>${config.label}参考视频</strong></div>${reference ? `<button data-locate="${reference.id}">定位拆分节点</button>` : ""}</div>
            ${preview}
            ${reference ? this.fieldHtml(reference, "trim_start", "参考截取开始（秒）") + this.fieldHtml(reference, "trim_end", "参考截取结束（秒，0为结尾）") : ""}
            <label class="h3te-field"><span>${reference ? "替换参考视频" : "上传参考视频"}</span><input type="file" accept="video/*" data-reference-file="${node.id}"></label>
            <button data-attach-reference="${node.id}">${reference ? "上传并替换连接" : "上传并创建参考节点"}</button>
        </section>`;
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
                const loader = this.createNode(TYPES.loadVideo, [clip.pos[0] - 620, clip.pos[1]], { file: path });
                const reference = this.createNode(TYPES.motionReference, [clip.pos[0] - 310, clip.pos[1]], {
                    trim_start: 0,
                    trim_end: Math.min(15, Math.max(0.05, Number(widgetValue(clip, "end_time", 1)) - Number(widgetValue(clip, "start_time", 0)))),
                });
                if (!connectNodes(loader, loader.outputs?.[0]?.name, reference, "video")) throw new Error("无法连接加载视频与语义拆分节点");
                const inputIndex = slotIndex(clip, config.input);
                if (inputIndex < 0 || !reference.outputs?.[config.output]) throw new Error("片段不支持所选参考视频语义");
                reference.connect(config.output, clip, inputIndex);
                loader.properties ||= {};
                loader.properties.minimax_h3_reference_upload = true;
                reference.properties ||= {};
                reference.properties.minimax_h3_reference_upload = true;
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
        const libraryTab = event.target.closest("[data-library-tab]");
        if (libraryTab) {
            this.libraryTab = libraryTab.dataset.libraryTab;
            this.renderLibrary();
            return;
        }
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
        const deleteReference = event.target.closest("[data-delete-reference-asset]");
        if (deleteReference) return this.deleteReferenceAsset(deleteReference.dataset.deleteReferenceAsset);
        const action = event.target.closest("[data-action]")?.dataset.action;
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
        if (action === "refresh") return this.render();
        if (action === "save") return this.saveWorkflow(true);
        if (action === "bootstrap") return this.bootstrapProject();
        if (action === "layout") {
            const changed = this.layoutProjectNodes(true);
            this.setStatus(changed ? "已按人物、环境、轨道和参考媒体自动排布节点" : "当前工程节点已经排布整齐");
            return;
        }
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
        const saveActorCard = event.target.closest("[data-save-actor-card]");
        if (saveActorCard) return this.saveActorAsResource(app.graph.getNodeById(Number(saveActorCard.dataset.saveActorCard)));

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
        const field = event.target.closest("[data-field-node]");
        if (!field) return;
        const node = app.graph.getNodeById(Number(field.dataset.fieldNode));
        const current = widget(node, field.dataset.fieldName)?.value;
        const value = typeof current === "number" ? Number(field.value) : field.value;
        if (setWidgetValue(node, field.dataset.fieldName, value)) {
            this.data = collectTimeline(this.currentTimeline());
            if (["start_time", "end_time", "action_type", "name"].includes(field.dataset.fieldName)) {
                this.renderLibrary();
                this.renderTracks();
            }
            this.setStatus("已同步到工作流");
        }
    }

    onPointerDown(event) {
        const clipElement = event.target.closest(".h3te-clip");
        if (clipElement) {
            event.preventDefault();
            const node = app.graph.getNodeById(Number(clipElement.dataset.nodeId));
            const clip = this.data.tracks.flatMap((track) => track.clips).find((item) => item.node === node);
            if (!clip) return;
            this.selection = { kind: "clip", node, nodes: [node], title: `${KIND_NAMES[clip.kind] || clip.kind}片段${clip.resultNode ? " · 已缓存" : ""}` };
            this.renderInspector();
            if (this.libraryTab === "media") this.renderLibrary();
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
        const ruler = event.target.closest(".h3te-ruler");
        if (ruler && this.data) {
            const rect = ruler.getBoundingClientRect();
            this.playhead = Math.max(0, Math.min(this.data.duration, (event.clientX - rect.left) / this.pixelsPerSecond));
            this.renderTracks();
            this.updateTime();
        }
    }

    onPointerMove(event) {
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

    onPointerUp() {
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

    togglePlay() {
        if (!this.data) return;
        this.playing = !this.playing;
        $("[data-action=play]", this.root).textContent = this.playing ? "❚❚" : "▶";
        if (!this.playing) return;
        if (this.playhead >= this.data.duration) this.playhead = 0;
        let previous = performance.now();
        const frame = (now) => {
            if (!this.playing) return;
            this.playhead += (now - previous) / 1000;
            previous = now;
            if (this.playhead >= this.data.duration) {
                this.playhead = this.data.duration;
                this.playing = false;
                $("[data-action=play]", this.root).textContent = "▶";
            }
            this.root.querySelectorAll(".h3te-playhead").forEach((line) => line.style.left = `${this.playhead * this.pixelsPerSecond}px`);
            this.updateTime();
            if (this.playing) requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
    }

    updateTime() {
        const minutes = Math.floor(this.playhead / 60);
        const seconds = this.playhead - minutes * 60;
        $("[data-role=time]", this.root).textContent = `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
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
        if (nodeData.name !== TYPES.timeline) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const button = this.addWidget("button", "打开导演时间轴", null, () => openEditor(this));
            button.serialize = false;
        };
    },
});
