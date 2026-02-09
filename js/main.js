import { app } from "/scripts/app.js";


function displayError(category, message) {
    app.extensionManager.toast.add({
        severity: "error",
        summary: category + " error",
        detail: message,
        life: 5000
    });
}


function h(tag, f) {
    const x = document.createElement(tag);
    f(x);
    return x;
}


function cleanup(text) {
    // Replaces tabs with space
    text = text.replace(/\t+/g, " ");

/*
    // Normalize to use \n for newlines
    text = text.replace(/\r/g, "\n");

    // Remove unnecessary spaces, periods, and commas before a weight
    text = text.replace(/[\., ]*: *([\d\.]+) *\),?/g, ":$1)");

    // Remove unnecessary spaces, periods, and commas at the beginning of a line
    text = text.replace(/(?:^|\n)[\., ]+/g, "\n");

    // Remove unnecessary spaces at the end of a line
    text = text.replace(/[ ]+\n/g, "\n");

    // Remove unnecessary periods and commas
    text = text.replace(/([\.,])[\., ]+/g, "$1 ");

    // Remove unnecessary newlines
    text = text.replace(/\n{3,}/g, "\n\n");*/

    text = text.trim();

    return text;
}


function cleanupPrompt(text) {
    // Remove unnecessary commas at the beginning and end
    text = text.replace(/(?:^[, ]+)|(?:[, ]+$)/g, "");

    // Remove unnecessary commas
    text = text.replace(/,[, ]+/g, ", ");

    // Adds a space after commas
    text = text.replace(/,(?! |$)/g, ", ");

    // Remove unnecessary spaces before a comma
    text = text.replace(/ +(?=,)/g, "");

    // Remove unnecessary spaces
    text = text.replace(/ {2,}/g, " ");

    return text;
}


class Bundle {
    constructor(name) {
        this.name = name.trim();
    }

    serialize() {
        return `BUNDLE: ${this.name}\n`;
    }

    render(root) {
        return h("div", (dom) => {
            dom.style.display = "flex";
            dom.style.flexDirection = "row";
            dom.style.alignItems = "center";

            dom.appendChild(h("span", (dom) => {
                dom.style.height = "1lh";
                dom.style.marginLeft = "5px";
                dom.style.color = "mediumorchid";

                dom.textContent = "BUNDLE: ";
            }));

            dom.appendChild(h("span", (dom) => {
                dom.style.height = "1lh";
                dom.style.flex = "1";
                dom.style.color = "darkorange";

                dom.textContent = this.name;
            }));
        });
    }
}


class Break {
    constructor() {}

    serialize() {
        return "---\n";
    }

    render(root) {
        return h("div", (dom) => {
            dom.style.height = "1lh";

            dom.style.display = "flex";
            dom.style.flexDirection = "row";
            dom.style.alignItems = "center";

            dom.appendChild(h("div", (dom) => {
                dom.style.width = "100%";
                dom.style.height = "2px";
                dom.style.border = "none";
                dom.style.margin = "0px";
                dom.style.backgroundColor = "hsl(207.3, 25%, 35%)";
            }));
        });
    }
}


class Blank {
    constructor() {}

    serialize() {
        return "\n";
    }

    render(root) {
        return h("br", (dom) => {
            dom.style.lineHeight = "1lh";
        });
    }
}


class Line {
    constructor(value) {
        const comment = /^ *(#|\/\/)?(.*)$/.exec(value);

        if (comment[1]) {
            this.enabled = false;

        } else {
            this.enabled = true;
        }

        const weight = /^(.*)\* *([\-\d\.]+) *$/.exec(comment[2]);

        if (weight) {
            this.weight = +((+weight[2]).toFixed(2));
            this.prompt = cleanupPrompt(weight[1]);

        } else {
            this.weight = 1.0;
            this.prompt = cleanupPrompt(comment[2]);
        }
    }

    serialize() {
        const weight = this.weight.toFixed(2);

        const enabled = this.enabled ? " " :  "#";

        if (weight === "1.00") {
            return `${enabled} ${this.prompt}\n`;

        } else {
            return `${enabled} ${this.prompt} * ${weight}\n`;
        }
    }

    render(root) {
        function weightAmount(event) {
            if (event.shiftKey) {
                return 0.10;
            } else if (event.ctrlKey) {
                return 0.01;
            } else {
                return 0.05;
            }
        }

        return h("div", (dom) => {
            let updateWeight;

            dom.style.display = "flex";
            dom.style.flexDirection = "row";
            dom.style.alignItems = "center";

            dom.appendChild(h("input", (dom) => {
                dom.setAttribute("type", "checkbox");

                dom.style.cursor = "pointer";

                dom.style.width = "16px";
                dom.style.height = "16px";
                dom.style.margin = "0px";

                if (this.enabled) {
                    dom.setAttribute("checked", "");
                }

                dom.addEventListener("change", () => {
                    this.enabled = dom.checked;
                    root.save();
                });
            }));

            dom.appendChild(h("span", (dom) => {
                dom.textContent = this.prompt;

                dom.style.height = "1lh";
                dom.style.flex = "1";
                dom.style.marginLeft = "6px";
            }));

            dom.appendChild(h("span", (dom) => {
                updateWeight = () => {
                    const weight = this.weight.toFixed(2);

                    dom.textContent = weight;

                    if (weight === "1.00" || weight === "-1.00") {
                        dom.style.opacity = "0.2";

                    } else {
                        dom.style.opacity = "";
                    }

                    if (weight === "0.00") {
                        dom.style.color = "white";

                    } else if (weight[0] === "-") {
                        dom.style.color = "hsl(0, 100%, 70%)";

                    } else {
                        dom.style.color = "hsl(120, 100%, 80%)";
                    }
                };

                updateWeight();

                dom.style.cursor = "pointer";

                dom.style.height = "1lh";
                dom.style.marginLeft = "6px";
                dom.style.marginRight = "6px";

                dom.addEventListener("click", () => {
                    this.weight = -this.weight;
                    updateWeight();
                    root.save();
                });
            }));

            dom.appendChild(h("button", (dom) => {
                dom.tabIndex = "-1";

                dom.style.width = "18px";
                dom.style.height = "18px";
                dom.style.margin = "0px";
                dom.style.marginRight = "1px";

                dom.style.cursor = "pointer";

                dom.style.display = "flex";
                dom.style.alignItems = "center";
                dom.style.justifyContent = "center";

                dom.appendChild(h("div", (dom) => {
                    dom.textContent = "-";
                }));

                dom.addEventListener("click", (event) => {
                    this.weight -= weightAmount(event);
                    updateWeight();
                    root.save();
                });
            }));

            dom.appendChild(h("button", (dom) => {
                dom.tabIndex = "-1";

                dom.style.width = "18px";
                dom.style.height = "18px";
                dom.style.margin = "0px";

                dom.style.cursor = "pointer";

                dom.style.display = "flex";
                dom.style.alignItems = "center";
                dom.style.justifyContent = "center";

                dom.appendChild(h("div", (dom) => {
                    dom.textContent = "+";
                }));

                dom.addEventListener("click", () => {
                    this.weight += weightAmount(event);
                    updateWeight();
                    root.save();
                });
            }));
        });
    }
}


class PromptToggle {
    static parseLines(value) {
        value = cleanup(value);

        const lines = [];

        value.split(/(?:\r\n|\n)/g).forEach((line) => {
            line = line.trim();

            if (line === "") {
                lines.push(new Blank());

            } else if (line === "BREAK" || /^\-{3,}$/.test(line)) {
                lines.push(new Break());

            } else if (line.startsWith("BUNDLE:")) {
                lines.push(new Bundle(line.slice("BUNDLE:".length)));

            } else {
                lines.push(new Line(line));
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

            dom.style.fontFamily = "monospace";

            dom.style.cursor = "default";
        });
    }

    replaceLines(value) {
        this.lines = PromptToggle.parseLines(value);
    }

    serialize() {
        return this.lines.map((line) => line.serialize()).join("");
    }

    save() {
        this.textWidget.value = this.serialize();
    }

    renderEditBox() {
        return h("div", (dom) => {
            dom.textContent = this.editText;

            // Textareas can't be dynamically resized, so we use contenteditable as a workaround
            dom.setAttribute("contenteditable", "plaintext-only");
            dom.setAttribute("placeholder", "Prompt...");

            // Automatically focus the textbox
            // We can't use autofocus because the textbox is dynamically generated
            queueMicrotask(() => {
                dom.focus();
            });

            dom.className = "comfy-multiline-input";

            dom.style.flex = "1";
            dom.style.cursor = "text";
            dom.style.border = "1px solid lightsteelblue";
            dom.style.padding = "3px 4px";
            dom.style.overflow = "auto";

            dom.style.caretColor = "crimson";

            dom.style.fontFamily = "inherit";
            dom.style.fontSize = "inherit";
            dom.style.lineHeight = "inherit";
            dom.style.minHeight = "5lh";
            dom.style.boxSizing = "content-box";
            dom.style.backgroundColor = "hsl(240, 100%, 13%)";
            dom.style.color = "white";
            dom.style.borderRadius = "5px";

            dom.addEventListener("input", (event) => {
                this.editText = dom.textContent;

                event.stopPropagation();
            });

            // These events are needed to stop ComfyUI from glitching out and causing problems
            dom.addEventListener("contextmenu", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("mousedown", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("mousemove", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("mouseup", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("pointerdown", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("pointermove", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("pointerup", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("wheel", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("copy", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("paste", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("keydown", (event) => {
                event.stopPropagation();
            });

            dom.addEventListener("keyup", (event) => {
                event.stopPropagation();
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
            dom.style.marginTop = "12px";

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
                dom.style.padding = "4px 0px";
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
            const textWidget = node.widgets[0];

            textWidget.options.hidden = true;

            const prompt = new PromptToggle(textWidget, textWidget.value);

            node.onConfigure = () => {
                prompt.replaceLines(textWidget.value);
                prompt.render();
            };

            prompt.render();

            node.addDOMWidget(
                "prompt_helpers_prompt_toggle",
                "prompt_helpers: PromptToggle",
                prompt.root,
            );
        }
    },
});
