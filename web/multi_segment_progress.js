import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const STAGE_LABELS = {
    conditioning: "准备提示词、参考素材与段间引导",
    noise: "准备噪声",
    guider: "准备引导器",
    scheduler: "准备采样计划",
    sampling: "模型采样",
    video_decode: "解码画面",
    audio_decode: "解码声音",
    trim: "裁切片段",
    result_prepare: "整理已生成结果（跳过采样）",
    segment_video: "封装当前片段",
    checkpoint: "保存当前片段",
    cache_load: "读取已完成片段",
    components: "读取片段音视频",
    join: "连续片段拼接",
    final_video: "封装最终视频",
};

const STAGE_PROGRESS = {
    conditioning: 0.03,
    noise: 0.05,
    guider: 0.07,
    scheduler: 0.09,
    sampling: 0.1,
    video_decode: 0.88,
    audio_decode: 0.92,
    trim: 0.95,
    result_prepare: 0.95,
    segment_video: 0.96,
    checkpoint: 0.97,
    cache_load: 0.97,
    components: 0.98,
    join: 0.99,
    final_video: 1.0,
};

const STAGE_ORDER = Object.keys(STAGE_LABELS);

function parseStage(nodeId) {
    const match = String(nodeId).match(/segment_(\d+)_of_(\d+)_(conditioning|noise|guider|scheduler|sampling|video_decode|audio_decode|trim|result_prepare|segment_video|checkpoint|cache_load|components|join|final_video)$/);
    return match ? { segment: Number(match[1]), total: Number(match[2]), stage: match[3] } : null;
}

function formatDuration(milliseconds) {
    const seconds = Math.max(0, milliseconds) / 1000;
    if (seconds < 60) {
        return `${seconds.toFixed(1)}秒`;
    }
    const minutes = Math.floor(seconds / 60);
    const remaining = seconds - minutes * 60;
    if (minutes < 60) {
        return `${minutes}分${remaining.toFixed(1)}秒`;
    }
    const hours = Math.floor(minutes / 60);
    return `${hours}时${minutes % 60}分${remaining.toFixed(1)}秒`;
}

function resetTiming(node, now = performance.now()) {
    node.segmentTiming = {
        startedAt: now,
        lastUpdateAt: now,
        activeKey: null,
        entries: [],
        entriesByKey: new Map(),
        completed: false,
    };
}

function entryKey(parsed) {
    return `${parsed.segment}:${parsed.stage}`;
}

function ensureEntry(timing, parsed, now) {
    const key = entryKey(parsed);
    let entry = timing.entriesByKey.get(key);
    if (!entry) {
        entry = { key, parsed, startedAt: now, finishedAt: null, value: 0, max: 0 };
        timing.entriesByKey.set(key, entry);
        timing.entries.push(entry);
    }
    return entry;
}

function finishActive(timing, now) {
    if (!timing.activeKey) {
        return;
    }
    const active = timing.entriesByKey.get(timing.activeKey);
    if (active && active.finishedAt === null) {
        active.finishedAt = now;
    }
    timing.activeKey = null;
}

function updateTiming(node, related, now) {
    const running = related.find(({ state }) => state.state === "running");
    if (!node.segmentTiming) {
        resetTiming(node, now);
    }
    const timing = node.segmentTiming;

    if (running) {
        const key = entryKey(running.parsed);
        if (timing.activeKey !== key) {
            finishActive(timing, now);
            const entry = ensureEntry(timing, running.parsed, now);
            entry.startedAt = now;
            entry.finishedAt = null;
            timing.activeKey = key;
        }
        const entry = timing.entriesByKey.get(key);
        entry.value = Number(running.state.value) || 0;
        entry.max = Number(running.state.max) || 0;
    }

    for (const item of related) {
        if (item.state.state !== "finished") {
            continue;
        }
        const entry = ensureEntry(timing, item.parsed, timing.lastUpdateAt);
        entry.value = Number(item.state.value) || entry.value;
        entry.max = Number(item.state.max) || entry.max;
        if (entry.finishedAt === null) {
            entry.finishedAt = now;
        }
        if (timing.activeKey === entry.key) {
            timing.activeKey = null;
        }
    }

    const finished = related.length > 0 && related.every(({ state }) => state.state === "finished");
    if (finished) {
        finishActive(timing, now);
        timing.completed = true;
    }
    timing.lastUpdateAt = now;
    return { timing, running, finished };
}

function progressValue(running) {
    if (!running) {
        return 1;
    }
    let stageProgress = STAGE_PROGRESS[running.parsed.stage] ?? 0;
    if (running.parsed.stage === "sampling" && running.state.max > 0) {
        const sampleProgress = Math.max(0, Math.min(1, running.state.value / running.state.max));
        stageProgress += sampleProgress * 0.76;
    }
    return (running.parsed.segment - 1 + stageProgress) / running.parsed.total;
}

function progressText(node, detail) {
    const states = Object.values(detail?.nodes ?? {});
    const related = states.map((state) => ({ state, parsed: parseStage(state.node_id) }))
        .filter(({ state, parsed }) => parsed && String(state.display_node_id) === String(node.id));
    if (!related.length) {
        return null;
    }

    const now = performance.now();
    const { timing, running, finished } = updateTiming(node, related, now);
    const overall = finished ? 100 : Math.min(99, Math.round(progressValue(running) * 100));
    const lines = [`总体 ${overall}% · 累计 ${formatDuration(now - timing.startedAt)}`];

    const entries = timing.entries.slice().sort((left, right) =>
        left.parsed.segment - right.parsed.segment
        || STAGE_ORDER.indexOf(left.parsed.stage) - STAGE_ORDER.indexOf(right.parsed.stage));
    for (const entry of entries) {
        const label = `片段 ${entry.parsed.segment}/${entry.parsed.total} · ${STAGE_LABELS[entry.parsed.stage]}`;
        if (entry.finishedAt !== null) {
            lines.push(`✓ ${label}：${formatDuration(entry.finishedAt - entry.startedAt)}`);
            continue;
        }
        let sampleDetail = "";
        if (entry.parsed.stage === "sampling" && entry.max > 0) {
            sampleDetail = ` ${entry.value}/${entry.max}`;
        }
        lines.push(`▶ ${label}${sampleDetail}：已耗时 ${formatDuration(now - entry.startedAt)}`);
    }

    if (finished) {
        lines.push(`✓ 已完成全部 ${related[0].parsed.total} 个片段`);
    }
    return lines.join("\n");
}

function stopStepPreview(node) {
    if (node.stepPreviewAnimation) {
        cancelAnimationFrame(node.stepPreviewAnimation);
        node.stepPreviewAnimation = null;
    }
}

function clearStepPreview(node, label = "等待采样预览") {
    stopStepPreview(node);
    node.stepPreviewToken = (node.stepPreviewToken ?? 0) + 1;
    node.stepPreviewFrames = [];
    if (node.stepPreviewStatus) {
        node.stepPreviewStatus.textContent = label;
    }
    const canvas = node.stepPreviewCanvas;
    if (canvas) {
        const context = canvas.getContext("2d");
        context.fillStyle = "#111";
        context.fillRect(0, 0, canvas.width, canvas.height);
    }
}

function drawPreviewFrame(canvas, image) {
    const context = canvas.getContext("2d");
    context.fillStyle = "#111";
    context.fillRect(0, 0, canvas.width, canvas.height);
    const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    context.drawImage(image, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
}

async function setStepPreview(node, detail) {
    const encoded = Array.isArray(detail?.frames) ? detail.frames : [];
    if (!encoded.length || !node.stepPreviewCanvas) {
        return;
    }
    const token = (node.stepPreviewToken ?? 0) + 1;
    node.stepPreviewToken = token;
    const frames = await Promise.all(encoded.map((value) => new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = `data:image/jpeg;base64,${value}`;
    })));
    if (node.stepPreviewToken !== token) {
        return;
    }

    stopStepPreview(node);
    node.stepPreviewFrames = frames;
    node.stepPreviewStatus.textContent = `片段 ${detail.segment_index}/${detail.segment_total} · 采样 ${detail.step}/${detail.total_steps} · 潜空间动态预览`;
    const duration = Math.max(0.1, Number(detail.duration_seconds) || 1) * 1000;
    const startedAt = performance.now();
    const animate = (now) => {
        const index = Math.floor(((now - startedAt) % duration) / duration * frames.length);
        drawPreviewFrame(node.stepPreviewCanvas, frames[index]);
        node.stepPreviewAnimation = requestAnimationFrame(animate);
    };
    node.stepPreviewAnimation = requestAnimationFrame(animate);
}

app.registerExtension({
    name: "MiniMaxH3.MultiSegmentProgress",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MiniMaxH3MultiSegmentGenerate") {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.segmentProgressWidget = ComfyWidgets.STRING(
                this,
                "详细进度与耗时",
                ["STRING", { multiline: true }],
                app,
            ).widget;
            this.segmentProgressWidget.inputEl.readOnly = true;
            this.segmentProgressWidget.inputEl.placeholder = "执行后显示每一步的进度和耗时";
            this.segmentProgressWidget.serializeValue = async () => "";
            this.segmentProgressWidget.value = "等待执行";
            const preview = document.createElement("div");
            preview.style.cssText = "display:flex;flex-direction:column;gap:6px;width:100%;height:100%;box-sizing:border-box;padding:4px;";
            const status = document.createElement("div");
            status.style.cssText = "color:var(--input-text,#ddd);font-size:12px;line-height:18px;";
            status.textContent = "等待采样预览";
            const canvas = document.createElement("canvas");
            canvas.width = 576;
            canvas.height = 324;
            canvas.style.cssText = "display:block;width:100%;height:auto;max-height:324px;background:#111;border:1px solid #444;border-radius:5px;object-fit:contain;";
            preview.append(status, canvas);
            this.stepPreviewStatus = status;
            this.stepPreviewCanvas = canvas;
            this.addDOMWidget("逐步视频预览", "preview", preview, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => 360,
            });
            this.addWidget("button", "停止生成并保留已完成片段", null, () => {
                void api.interrupt();
            });
            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                stopStepPreview(this);
                return onRemoved?.apply(this, arguments);
            };
            resetTiming(this);
            clearStepPreview(this);
            this.setSize([Math.max(this.size[0], 640), Math.max(this.size[1], 880)]);
        };
    },
});

api.addEventListener("execution_start", () => {
    for (const node of app.graph?._nodes ?? []) {
        if (node.type !== "MiniMaxH3MultiSegmentGenerate" || !node.segmentProgressWidget) {
            continue;
        }
        resetTiming(node);
        node.segmentProgressWidget.value = "准备执行";
        clearStepPreview(node, "准备采样");
    }
});

api.addEventListener("minimax_h3_step_video_preview", ({ detail }) => {
    const node = app.graph?.getNodeById(Number(detail?.node_id));
    if (node?.type === "MiniMaxH3MultiSegmentGenerate") {
        void setStepPreview(node, detail);
    }
});

api.addEventListener("progress_state", ({ detail }) => {
    for (const node of app.graph?._nodes ?? []) {
        if (node.type !== "MiniMaxH3MultiSegmentGenerate" || !node.segmentProgressWidget) {
            continue;
        }
        const text = progressText(node, detail);
        if (text) {
            node.segmentProgressWidget.value = text;
            app.graph.setDirtyCanvas(true, false);
        }
    }
});
