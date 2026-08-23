import { app } from "/scripts/app.js";

const schedulers = new Set([
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform",
    "beta", "normal", "linear_quadratic", "kl_optimal",
]);

function migrateGenerationJob(info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length < 3) {
        return;
    }

    const oldInput = info.inputs?.some((input) => input.name === "target_segment_seconds");
    const validCurrent = values.length === 10
        && ["fixed", "increment", "decrement", "randomize"].includes(values[3])
        && schedulers.has(values[4])
        && typeof values[5] === "number"
        && typeof values[6] === "number"
        && ["match", "max"].includes(values[7])
        && ["不输出", "输出 N/A"].includes(values[8])
        && typeof values[9] === "number";
    if (!oldInput && validCurrent) {
        return;
    }

    const schedulerIndex = values.reduce(
        (found, value, index) => index >= 3 && schedulers.has(value) ? index : found,
        -1,
    );
    const scheduler = schedulerIndex >= 0 ? values[schedulerIndex] : "simple";
    const following = schedulerIndex >= 0 ? values.slice(schedulerIndex + 1) : [];
    const steps = following.find((value) => Number.isInteger(value) && value >= 1) ?? 4;
    const stepsIndex = following.indexOf(steps);
    const denoise = following.slice(stepsIndex + 1).find(
        (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1,
    ) ?? 1.0;
    const refImageSize = values.find((value) => value === "match" || value === "max") ?? "match";
    const refIndex = values.indexOf(refImageSize);
    const afterRef = refIndex >= 0 ? values.slice(refIndex + 1) : [];
    const emptySections = afterRef.find((value) => value === "不输出" || value === "输出 N/A") ?? "不输出";
    const continuity = afterRef.find(
        (value) => typeof value === "number" && Number.isFinite(value) && value >= 0.21,
    ) ?? 0.92;
    info.widgets_values = [
        typeof values[0] === "number" && Number.isFinite(values[0]) ? values[0] : 0.98,
        typeof values[1] === "string" ? values[1] : "16:9",
        typeof values[2] === "number" && Number.isFinite(values[2]) ? values[2] : 0,
        "fixed",
        scheduler,
        steps,
        denoise,
        refImageSize,
        emptySections,
        continuity,
    ];
    if (Array.isArray(info.inputs)) {
        info.inputs = info.inputs.filter(
            (input) => !["target_segment_seconds", "handoff_tail_frames", "overlap_seconds"].includes(input.name),
        );
    }
}

function migrateGeneratedVideoSave(info) {
    const values = info?.widgets_values;
    if (Array.isArray(values)
        && values[0] === "video/MiniMax H3 Multi Segment"
        && values[2]
        && typeof values[2] === "object") {
        values[2] = values[2].codec ?? "auto";
    }
}

function migrateCharacterCard(info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length <= 4) {
        return;
    }
    const stylePriority = values.find((value) => value === "character" || value === "global") ?? "global";
    info.widgets_values = [values[0] ?? "the young woman", values[1] ?? "", stylePriority, values[8] ?? ""];
    if (Array.isArray(info.inputs)) {
        info.inputs = info.inputs.filter((input) => ![
            "preservation", "default_position", "default_pose", "default_emotion", "default_appearance",
        ].includes(input.name));
    }
}

function migrateEnvironmentCard(info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length <= 3) {
        return;
    }
    info.widgets_values = [values[0] ?? "the environment", values[1] ?? "", values[3] ?? ""];
    if (Array.isArray(info.inputs)) {
        info.inputs = info.inputs.filter((input) => ![
            "default_time_weather", "default_atmosphere", "preservation", "reference_usage",
        ].includes(input.name));
    }
}

function migrateReferenceSplit(info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length <= 2) {
        return;
    }
    const numeric = values.filter((value) => typeof value === "number" && Number.isFinite(value));
    info.widgets_values = [numeric.at(-2) ?? 0, numeric.at(-1) ?? 0];
    if (Array.isArray(info.inputs)) {
        info.inputs = info.inputs.filter((input) => !["role", "include_audio"].includes(input.name));
    }
}

app.registerExtension({
    name: "MiniMaxH3.GenerationJobMigration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MiniMaxH3MotionReference") {
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                migrateReferenceSplit(info);
                return configure?.apply(this, arguments);
            };
            return;
        }
        if (nodeData.name === "MiniMaxH3Character" || nodeData.name === "MiniMaxH3Environment") {
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                if (nodeData.name === "MiniMaxH3Character") {
                    migrateCharacterCard(info);
                } else {
                    migrateEnvironmentCard(info);
                }
                return configure?.apply(this, arguments);
            };
            return;
        }
        if (nodeData.name === "SaveVideo") {
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                migrateGeneratedVideoSave(info);
                return configure?.apply(this, arguments);
            };
            return;
        }
        if (nodeData.name !== "MiniMaxH3GenerationJob") {
            return;
        }
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            migrateGenerationJob(info);
            return configure?.apply(this, arguments);
        };
    },
});
