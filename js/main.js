import { app } from "/scripts/app.js";


function h(tag, f) {
    const x = document.createElement(tag);
    f(x);
    return x;
}


function cleanup(text) {
    // Normalize to use \n for newlines
    text = text.replace(/\r/g, "\n");

    // Remove unnecessary spaces, periods, and commas before a weight
    text = text.replace(/[\., \t]*:[ \t]*([\d\.]+)[ \t]*\),?/g, ":$1)");

    // Remove unnecessary spaces, periods, and commas at the beginning of a line
    text = text.replace(/(?:^|\n)[\., \t]+/g, "\n");

    // Remove unnecessary spaces at the end of a line
    text = text.replace(/[ \t]+\n/g, "\n");

    // Remove unnecessary periods and commas
    text = text.replace(/([\.,])[\., \t]+/g, "$1 ");

    // Remove unnecessary newlines
    text = text.replace(/\n{3,}/g, "\n\n");

    text = text.trim();

    return text;
}


function cleanupPrompt(text) {
    // Remove unnecessary periods and commas at the beginning and end
    text = text.replace(/(?:^[\., \t]+)|(?:[\., \t]+$)/g, "");

    // Remove unnecessary periods and commas
    text = text.replace(/([\.,])[\., \t]+/g, "$1 ");

    // Remove unnecessary spaces
    text = text.replace(/ {2,}/g, " ");

    return text;
}


class Break {
    constructor(index) {
        this.index = index;
    }

    serialize() {
        return "BREAK";
    }

    render(root) {
        return h("hr", (dom) => {});
    }
}


class Blank {
    constructor(index) {
        this.index = index;
    }

    serialize() {
        return "";
    }

    render(root) {
        return h("br", (dom) => {});
    }
}


class Line {
    constructor(index, value) {
        this.index = index;

        const comment = /^(\/\/)?(.*)$/.exec(value);

        if (comment[1]) {
            this.checked = false;

        } else {
            this.checked = true;
        }

        const weight = /^[ \t]*\((.*):[ \t]*([\d\.]+)[ \t]*\)[\.,]*$/.exec(comment[2]);

        if (weight) {
            this.weight = +weight[2];
            this.prompt = cleanupPrompt(weight[1]);

        } else {
            this.weight = 1.0;
            this.prompt = cleanupPrompt(comment[2]);
        }
    }

    serialize() {
        if (this.weight === 1.0) {
            return `${this.checked ? "   " :  "// "} ${this.prompt},`;

        } else {
            return `${this.checked ? "   " :  "// "}(${this.prompt}: ${this.weight.toFixed(2)}),`;
        }
    }

    render(root) {
        return h("div", (dom) => {
            dom.style.display = "flex";
            dom.style.flexDirection = "row";

            dom.appendChild(h("input", (dom) => {
                dom.setAttribute("type", "checkbox");

                if (this.checked) {
                    dom.setAttribute("checked", "");
                }

                dom.addEventListener("change", () => {
                    this.checked = dom.checked;
                    root.save();
                });
            }));

            dom.appendChild(h("span", (dom) => {
                dom.textContent = this.prompt;

                dom.style.flex = "1";
                dom.style.marginLeft = "6px";
            }));

            dom.appendChild(h("span", (dom) => {
                dom.textContent = this.weight.toFixed(2);

                dom.style.marginRight = "6px";
            }));

            dom.appendChild(h("button", (dom) => {
                dom.style.marginRight = "1px";

                dom.tabIndex = "-1";

                dom.style.width = "18px";
                dom.style.height = "18px";

                dom.style.cursor = "pointer";

                dom.style.display = "flex";
                dom.style.alignItems = "center";
                dom.style.justifyContent = "center";
                dom.style.verticalAlign = "middle";

                dom.appendChild(h("span", (dom) => {
                    dom.textContent = "-";
                }));

                dom.addEventListener("click", () => {
                    this.weight -= 0.05;
                    root.save();
                    root.render();
                });
            }));

            dom.appendChild(h("button", (dom) => {
                dom.tabIndex = "-1";

                dom.style.width = "18px";
                dom.style.height = "18px";

                dom.style.cursor = "pointer";

                dom.style.display = "flex";
                dom.style.alignItems = "center";
                dom.style.justifyContent = "center";
                dom.style.verticalAlign = "middle";

                dom.appendChild(h("span", (dom) => {
                    dom.textContent = "+";
                }));

                dom.addEventListener("click", () => {
                    this.weight += 0.05;
                    root.save();
                    root.render();
                });
            }));
        });
    }
}


class PromptToggle {
    static parseLines(value) {
        const lines = [];

        value.split(/(?:\r\n|\n)/g).forEach((line, index) => {
            line = line.trim();

            if (line === "BREAK") {
                lines.push(new Break(index));

            } else if (line === "") {
                lines.push(new Blank(index));

            } else {
                lines.push(new Line(index, line));
            }
        });

        return lines;
    }

    constructor(textWidget, value) {
        this.textWidget = textWidget;

        this.lines = PromptToggle.parseLines(value);

        this.editing = false;
        this.editText = null;

        this.root = h("div", (dom) => {
            dom.style.display = "flex";
            dom.style.flexDirection = "column";

            dom.style.whiteSpace = "pre-wrap";
            dom.style.overflowWrap = "anywhere";
        });
    }

    replaceLines(value) {
        this.lines = PromptToggle.parseLines(value);
    }

    serialize() {
        return this.lines.map((line) => line.serialize()).join("\n");
    }

    save() {
        this.textWidget.value = this.serialize();
    }

    renderEditBox() {
        return h("textarea", (dom) => {
            dom.value = this.editText;

            dom.style.display = "block";
            dom.style.flex = "1";

            dom.style.fontFamily = "inherit";
            dom.style.fontSize = "inherit";
            dom.style.lineHeight = "inherit";
            dom.style.width = "100%";
            dom.style.border = "1px solid gainsboro";
            dom.style.borderRadius = "3px";
            dom.style.margin = "0px";
            dom.style.padding = "6px 8px";
            dom.style.overflow = "auto";
            dom.style.resize = "none";

            dom.rows = this.lines.length;

            dom.addEventListener("input", () => {
                this.editText = dom.value;
            });
        });
    }

    renderEditButton() {
        return h("button", (dom) => {
            if (this.editing) {
                dom.textContent = "💾 Save prompt";
            } else {
                dom.textContent = "📝 Edit prompt";
            }

            dom.style.cursor = "pointer";
            dom.style.padding = "6px 8px";
            dom.style.marginTop = "18px";

            if (this.editing) {
                dom.style.color = "springgreen";
            }

            dom.addEventListener("click", () => {
                if (this.editing) {
                    const text = this.editText;

                    this.editing = false;
                    this.editText = null;

                    this.replaceLines(text);
                    this.save();

                } else {
                    this.editing = true;
                    this.editText = this.serialize();
                }

                this.render();
            });
        });
    }

    render() {
        this.root.innerHTML = "";

        if (this.editing) {
            this.root.appendChild(this.renderEditBox());

        } else {
            this.root.appendChild(h("div", (dom) => {
                dom.style.flex = "1";
                dom.style.padding = "5px 0px";
                dom.style.overflow = "auto";

                this.lines.forEach((line) => {
                    dom.appendChild(line.render(this));
                });
            }));
        }

        this.root.appendChild(this.renderEditButton());
    }
}


app.registerExtension({
    name: "prompt_helpers: PromptToggle",
    nodeCreated(node) {
        if (node.comfyClass === "prompt_helpers: PromptToggle") {
            console.log(node);

            node.multiline = true;

            const textWidget = node.widgets[0];

            textWidget.options.hidden = true;

            const widget = new PromptToggle(textWidget, textWidget.options.getValue());

            node.onConfigure = () => {
                widget.replaceLines(textWidget.value);
                widget.render();
            };

            widget.render();

            node.addDOMWidget(
                "prompt_helpers_prompt_toggle",
                "prompt_helpers: PromptToggle",
                widget.root,
            );
        }
    }
});
