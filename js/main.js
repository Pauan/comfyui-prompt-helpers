// @TODO It should pan when using the middle mouse button on the node
import { app } from "/scripts/app.js";

{
    const link = document.createElement("link");
    link.href = "extensions/prompt_helpers/main.css";
    link.rel = "stylesheet";
    document.head.appendChild(link);
}


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
            dom.className = "bundle";

            dom.appendChild(h("span", (dom) => {
                dom.className = "bundle-header";
                dom.textContent = "BUNDLE: ";
            }));

            dom.appendChild(h("span", (dom) => {
                dom.className = "bundle-text";
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
            dom.className = "break";

            dom.appendChild(h("div", (dom) => {
                dom.className = "break-line";
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
            dom.className = "blank";
        });
    }
}


class Line {
    constructor(value, even) {
        this.even = even;

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

            dom.className = "line";
            dom.classList.toggle("even", this.even);

            dom.appendChild(h("input", (dom) => {
                dom.className = "line-checkbox";

                dom.setAttribute("type", "checkbox");

                if (this.enabled) {
                    dom.setAttribute("checked", "");
                }

                dom.addEventListener("change", () => {
                    this.enabled = dom.checked;
                    updateWeight();
                    root.save();
                });
            }));

            dom.appendChild(h("span", (dom) => {
                dom.className = "line-prompt";
                dom.textContent = this.prompt;
            }));

            dom.appendChild(h("span", (dom) => {
                dom.className = "line-weight";

                updateWeight = () => {
                    this.weight = +(this.weight.toFixed(2));

                    dom.textContent = this.weight.toFixed(2);

                    const disabled = this.weight === 0 || !this.enabled;

                    dom.classList.toggle("disabled", disabled);
                    dom.classList.toggle("negative", !disabled && this.weight < 0);
                    dom.classList.toggle("less", !disabled && this.weight > 0 && this.weight < 1);
                    dom.classList.toggle("normal", !disabled && this.weight === 1);
                    dom.classList.toggle("more", !disabled && this.weight > 1);
                };

                updateWeight();

                dom.addEventListener("click", () => {
                    this.weight = -this.weight;
                    updateWeight();
                    root.save();
                });
            }));

            dom.appendChild(h("button", (dom) => {
                dom.className = "line-button minus";

                dom.tabIndex = "-1";

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
                dom.className = "line-button";

                dom.tabIndex = "-1";

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

        let even = true;

        value.split(/(?:\r\n|\n)/g).forEach((line) => {
            line = line.trim();

            if (line === "") {
                lines.push(new Blank());

            } else if (line === "BREAK" || /^\-{3,}$/.test(line)) {
                lines.push(new Break());
                even = true;

            } else if (line.startsWith("BUNDLE:")) {
                lines.push(new Bundle(line.slice("BUNDLE:".length)));
                even = true;

            } else {
                lines.push(new Line(line, even));
                even = !even;
            }
        });

        return lines;
    }

    constructor(textWidget, value) {
        this.textWidget = textWidget;

        this.lines = PromptToggle.parseLines(value);

        this.editing = false;
        this.editText = null;

        this.holdingAlt = false;

        this.root = h("div", (dom) => {
            dom.className = "prompt_helpers-root";

            // @TODO MacOS support
            function altKey(event) {
                return !event.shiftKey && !event.ctrlKey && event.altKey && !event.metaKey;
            }

            const toggleCursor = (event) => {
                this.holdingAlt = altKey(event);
                this.updateCursor();
            };

            addEventListener("keydown", toggleCursor, true);
            addEventListener("keyup", toggleCursor, true);

            dom.addEventListener("click", (event) => {
                if (event.button === 0 && altKey(event)) {
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();

                    this.toggleMode();
                }
            }, true);
        });
    }

    updateCursor() {
        this.root.classList.toggle("cursor-save", this.holdingAlt && this.editing);
        this.root.classList.toggle("cursor-edit", this.holdingAlt && !this.editing);
    }

    replaceLines(value) {
        this.lines = PromptToggle.parseLines(value);
    }

    serialize() {
        return this.lines.map((line) => line.serialize()).join("");
    }

    save() {
        this.textWidget.value = this.serialize();
        this.textWidget.callback(this.textWidget.value);
    }

    toggleMode() {
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

        this.updateCursor();
        this.render();
    }

    renderEditBox() {
        return h("div", (dom) => {
            dom.className = "edit-box";

            dom.textContent = this.editText;

            // Textareas can't be dynamically resized, so we use contenteditable as a workaround
            dom.setAttribute("contenteditable", "plaintext-only");
            dom.setAttribute("placeholder", "Prompt...");

            // Automatically focus the textbox
            // We can't use autofocus because the textbox is dynamically generated
            queueMicrotask(() => {
                dom.focus();
            });

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
            dom.className = "toggle-mode";
            dom.classList.toggle("editing", this.editing);

            if (this.editing) {
                dom.textContent = "💾 Save prompt";
            } else {
                dom.textContent = "📝 Edit prompt";
            }

            dom.addEventListener("click", () => {
                this.toggleMode();
            });
        });
    }

    render() {
        this.root.innerHTML = "";

        if (this.editing) {
            this.root.appendChild(this.renderEditBox());

        } else {
            this.root.appendChild(h("div", (dom) => {
                dom.className = "lines";

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
