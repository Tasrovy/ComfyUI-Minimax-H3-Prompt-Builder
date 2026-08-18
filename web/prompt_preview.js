import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "MiniMaxH3.FinalPromptPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MiniMaxH3PromptPreview") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.promptPreviewWidget = ComfyWidgets.STRING(
                this,
                "最终提示词",
                ["STRING", { multiline: true }],
                app,
            ).widget;
            this.promptPreviewWidget.inputEl.readOnly = true;
            this.promptPreviewWidget.inputEl.placeholder = "执行节点后在这里显示每个生成片段的最终提示词";
            this.promptPreviewWidget.serializeValue = async () => "";
            this.setSize([Math.max(this.size[0], 620), Math.max(this.size[1], 420)]);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            this.promptPreviewWidget.value = message.text?.[0] ?? "";
            app.graph.setDirtyCanvas(true, false);
        };
    },
});
