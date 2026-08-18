import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const STAGE_LABELS = {
    conditioning: "准备提示词与参考素材",
    noise: "准备噪声",
    guider: "准备引导器",
    scheduler: "准备采样计划",
    sampling: "模型采样",
    video_decode: "解码画面",
    audio_decode: "解码声音",
    trim: "裁切片段",
    continuity: "提取连续性尾部",
    join: "重叠匹配与拼接",
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
    continuity: 0.97,
    join: 0.99,
    final_video: 1.0,
};

function parseStage(nodeId) {
    const match = String(nodeId).match(/segment_(\d+)_of_(\d+)_(conditioning|noise|guider|scheduler|sampling|video_decode|audio_decode|trim|continuity|join|final_video)$/);
    return match ? { segment: Number(match[1]), total: Number(match[2]), stage: match[3] } : null;
}

function progressText(node, detail) {
    const states = Object.values(detail?.nodes ?? {});
    const related = states.map((state) => ({ state, parsed: parseStage(state.node_id) }))
        .filter(({ state, parsed }) => parsed && String(state.display_node_id) === String(node.id));
    if (!related.length) {
        return null;
    }
    const totalSegments = related[0].parsed.total;
    const running = related.find(({ state }) => state.state === "running");
    if (!running) {
        const finished = related.every(({ state }) => state.state === "finished");
        return finished ? `已完成全部 ${totalSegments} 个片段` : null;
    }
    const { state, parsed } = running;
    let stageProgress = STAGE_PROGRESS[parsed.stage] ?? 0;
    let stageDetail = "";
    if (parsed.stage === "sampling" && state.max > 0) {
        const sampleProgress = Math.max(0, Math.min(1, state.value / state.max));
        stageProgress += sampleProgress * 0.76;
        stageDetail = ` ${state.value}/${state.max}`;
    }
    const overall = Math.min(99, Math.round(((parsed.segment - 1 + stageProgress) / totalSegments) * 100));
    return `片段 ${parsed.segment}/${totalSegments} · ${STAGE_LABELS[parsed.stage]}${stageDetail} · 总体 ${overall}%`;
}

app.registerExtension({
    name: "MiniMaxH3.MultiSegmentProgress",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3MultiSegmentGenerate") {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.segmentProgressWidget = this.addWidget("text", "当前进度", "等待执行", null, { serialize: false });
            this.segmentProgressWidget.disabled = true;
        };
    },
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
