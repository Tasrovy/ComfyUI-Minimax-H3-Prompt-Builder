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
    result_prepare: "整理已生成结果（跳过采样）",
    segment_video: "封装未裁剪片段",
    checkpoint: "保存未裁剪片段",
    cache_load: "读取已完成片段",
    components: "读取片段音视频",
    trim: "裁切时间轴可见部分",
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
    result_prepare: 0.92,
    segment_video: 0.93,
    checkpoint: 0.95,
    cache_load: 0.95,
    components: 0.96,
    trim: 0.97,
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
            this.addWidget("button", "停止生成并保留已完成片段", null, () => {
                void api.interrupt();
            });
            resetTiming(this);
            this.setSize([Math.max(this.size[0], 640), Math.max(this.size[1], 520)]);
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
