from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import os.path
import functools
import re
import glob
import torch
import yaml
import folder_paths
from json import (dumps)


class Pixels:
    def __init__(self, pixels):
        self.pixels = pixels

    def evaluate_int(self, value):
        return self.pixels


class Percent:
    def __init__(self, percent):
        self.percent = percent

    def evaluate_int(self, value):
        return int(round(self.percent * value))


class Region:
    def __init__(self, x, y, width, height, strength, feather, isolated):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.strength = strength
        self.feather = feather
        self.isolated = isolated

    def evaluate_feather(self, width, height):
        return (
            max(self.feather.evaluate_int(width), 0),
            max(self.feather.evaluate_int(height), 0),
        )

    @staticmethod
    def parse_int(json):
        if isinstance(json, int):
            return Pixels(json)

        elif isinstance(json, dict) and "percent" in json:
            return Percent(json["percent"])

        else:
            raise RuntimeError("Number must be an int or percent")

    @classmethod
    def from_json(cls, json):
        return Region(
            cls.parse_int(json["x"]),
            cls.parse_int(json["y"]),
            cls.parse_int(json["width"]),
            cls.parse_int(json["height"]),
            json.get("strength", 1.0),
            cls.parse_int(json.get("feather", 0)),
            json.get("isolated", False),
        )


class MaskRegion:
    def __init__(self, name, strength, isolated):
        self.name = name
        self.strength = strength
        self.isolated = isolated

    @classmethod
    def from_json(cls, json):
        return MaskRegion(
            json["name"],
            json.get("strength", 1.0),
            json.get("isolated", False),
        )


@io.comfytype(io_type="EZ_JSON")
class JSON(io.ComfyTypeIO):
    Type = list

    class Input(io.WidgetInput):
        '''JSON input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                     default: list=None, socketless: bool=None, force_input: bool=None, extra_dict=None, raw_link: bool=None, advanced: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input, extra_dict, raw_link, advanced)
            self.default: list

        def as_dict(self):
            return super().as_dict()


class ProcessState:
    def __init__(self):
        self.seen_prompts = set()
        self.bundles = {}
        self.chunks = []


    @staticmethod
    def cleanup_prompt(prompt):
        # Replace tabs with a space
        prompt = re.sub(r'\t+', r' ', prompt)

        # Replace _ with a space
        prompt = re.sub(r'_', r' ', prompt)

        # Replace newlines with a comma
        prompt = re.sub(r'[\n\r]+', r', ', prompt)

        # Removes commas and spaces at the start and end
        prompt = re.sub(r'(?:^[, ]+)|(?:[, ]+$)', r'', prompt)

        # Removes repeated commas
        prompt = re.sub(r',[, ]+', r', ', prompt)

        # Removes spaces before a comma
        prompt = re.sub(r' +(?=,)', r'', prompt)

        # Adds a space after commas
        prompt = re.sub(r',(?! )', r', ', prompt)

        # Removes repeated spaces
        prompt = re.sub(r' {2,}', r' ', prompt)

        # Replaces ( with \\(
        #prompt = re.sub(r'(?<!\\)\(', r'\\(', prompt)

        # Replaces ) with \\)
        #prompt = re.sub(r'(?<!\\)\)', r'\\)', prompt)

        return prompt


    def process_bundle(self, seen_bundles, outer_weight, item, name, chunk, prompts):
        if item.get("enabled", True):
            if not name in self.bundles:
                raise RuntimeError("Bundle {} not found.".format(name))

            if name in seen_bundles:
                raise RuntimeError("Infinite recursion when inserting bundle {}".format(name))

            bundle = self.bundles[name]

            self.process_children(
                seen_bundles.union({ name }),
                outer_weight * item.get("weight", 1.0),
                bundle["children"],
                chunk,
                prompts,
            )


    def process_children(self, seen_bundles, outer_weight, children, chunk, prompts):
        for item in children:
            if "prompt" in item:
                if item.get("enabled", True):
                    prompt = ProcessState.cleanup_prompt(item["prompt"])

                    if prompt != "":
                        key = (id(chunk["region"]), id(chunk["mask-region"]), prompt)

                        if key in self.seen_prompts:
                            raise RuntimeError("Duplicate prompt: {}".format(prompt))
                        else:
                            self.seen_prompts.add(key)

                        prompts.append({
                            "prompt": prompt,
                            "weight": round(outer_weight * item.get("weight", 1.0), 2),
                        })

            elif "lora" in item:
                if item.get("enabled", True):
                    prompts.append({
                        "lora": item["lora"],
                        "weight": round(outer_weight * item.get("weight", 1.0), 2),
                    })

            elif "bundle" in item:
                sub_prompts = []

                chunk["chunks"].append(sub_prompts)

                self.process_bundle(seen_bundles, outer_weight, item, item["bundle"], chunk, sub_prompts)

            elif "bundle-inline" in item:
                self.process_bundle(seen_bundles, outer_weight, item, item["bundle-inline"], chunk, prompts)

            else:
                raise RuntimeError("Unknown type.")


    def process(self, json):
        if not isinstance(json, list):
            raise RuntimeError("JSON is not an array.")

        for item in json:
            if "bundles" in item:
                if not isinstance(item["bundles"], list):
                    raise RuntimeError("Bundles is not an array.")

                for bundle in item["bundles"]:
                    if bundle.get("enabled", True):
                        name = bundle["name"]

                        if name in self.bundles:
                            raise RuntimeError("Duplicate bundle: {}".format(name))

                        else:
                            self.bundles[name] = bundle

            elif "chunk" in item:
                if not isinstance(item["chunk"], list):
                    raise RuntimeError("Chunk is not an array.")

                prompts = []

                chunk = {
                    "chunks": [prompts],
                    "region": item.get("region", None),
                    "mask-region": item.get("mask-region", None),
                }

                self.chunks.append(chunk)

                self.process_children(frozenset(), 1.0, item["chunk"], chunk, prompts)

            else:
                raise RuntimeError("Root must be either bundles or chunk.")

        return self.chunks


class ProcessJson(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: ProcessJson",
            display_name="Process JSON",
            description="Processes and normalizes JSON.",
            inputs=[
                JSON.Input("json"),
            ],
            outputs=[
                JSON.Output(display_name="JSON"),
            ],
        )


    """
        Expands bundles, normalizing all weights.
    """
    @classmethod
    def process_json(cls, json):
        state = ProcessState()
        return state.process(json)


    @staticmethod
    def iter_chunks(json):
        for chunk in json:
            region = chunk.get("region", None)
            mask_region = chunk.get("mask-region", None)
            positive_chunks = []
            negative_chunks = []

            for prompts in chunk["chunks"]:
                positive = []
                negative = []

                for item in prompts:
                    if "prompt" in item:
                        if item["weight"] > 0.0:
                            positive.append(item)

                        elif item["weight"] < 0.0:
                            item = item.copy()
                            item["weight"] = -item["weight"]
                            negative.append(item)

                if len(positive) > 0:
                    positive_chunks.append(positive)

                if len(negative) > 0:
                    negative_chunks.append(negative)

            if len(positive_chunks) > 0 or len(negative_chunks) > 0:
                if region is not None:
                    region = Region.from_json(region)

                if mask_region is not None:
                    mask_region = MaskRegion.from_json(mask_region)

                yield (positive_chunks, negative_chunks, region, mask_region)


    """
        Filters JSON to only have positive weights.
    """
    @classmethod
    def only_positive(cls, json):
        chunks = []

        for chunk in json:
            region = chunk.get("region", None)
            mask_region = chunk.get("mask-region", None)

            new_chunks = []

            for prompts in chunk["chunks"]:
                new_prompts = []

                for item in prompts:
                    if "prompt" in item:
                        if item["weight"] > 0.0:
                            new_prompts.append(item)

                if len(new_prompts) > 0:
                    new_chunks.append(new_prompts)

            if len(new_chunks) > 0:
                chunks.append({
                    "chunks": new_chunks,
                    "region": region,
                    "mask-region": mask_region,
                })

        return chunks


    """
        Filters JSON to only have negative weights.
    """
    @classmethod
    def only_negative(cls, json):
        chunks = []

        for chunk in json:
            region = chunk.get("region", None)
            mask_region = chunk.get("mask-region", None)

            new_chunks = []

            for prompts in chunk["chunks"]:
                new_prompts = []

                for item in prompts:
                    if "prompt" in item:
                        if item["weight"] < 0.0:
                            item = item.copy()
                            item["weight"] = -item["weight"]
                            new_prompts.append(item)

                if len(new_prompts) > 0:
                    new_chunks.append(new_prompts)

            if len(new_chunks) > 0:
                chunks.append({
                    "chunks": new_chunks,
                    "region": region,
                    "mask-region": mask_region,
                })

        return chunks


    """
        Converts list of prompts into text strings, ready to be CLIP encoded.
    """
    @staticmethod
    def serialize_prompts(prompts):
        text = []

        for item in prompts:
            prompt = item["prompt"] + ","
            weight = item["weight"]

            if weight == 1.0:
                text.append(prompt)
            else:
                text.append("({}:{}),".format(prompt, weight))

        if len(text) > 0:
            return " ".join(text)


    """
        Converts chunks into text strings, ready to be CLIP encoded.
    """
    @classmethod
    def serialize_chunks(cls, json):
        chunks = []

        for chunk in json:
            for prompts in chunk["chunks"]:
                prompt = cls.serialize_prompts(item for item in prompts if "prompt" in item)

                if prompt is not None:
                    chunks.append(prompt)

        return chunks


    @classmethod
    def execute(cls, json) -> io.NodeOutput:
        return io.NodeOutput(cls.process_json(json))


class ParseLines(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: ParseLines",
            display_name="Parse Lines",
            category="prompt_helpers/prompt",
            description="Parses text into JSON.",
            inputs=[
                io.String.Input("text", dynamic_prompts=False, multiline=True),
            ],
            outputs=[
                JSON.Output(display_name="JSON"),
            ],
        )


    @classmethod
    def fingerprint_inputs(cls, text):
        if "FILE:" in text:
            # @TODO Hack that causes ComfyUI to always execute the node
            # https://github.com/Comfy-Org/ComfyUI/discussions/12546
            return float("nan")

        else:
            return text


    @staticmethod
    def get_paths(path, root):
        # TODO include_hidden=True
        paths = glob.glob(path, root_dir=root, recursive=True)

        if len(paths) == 0:
            raise RuntimeError("File not found: {}".format(path))
        else:
            return [os.path.realpath(os.path.join(root, path), strict=True) for path in paths]


    @staticmethod
    def parse_object(object):
        match = re.fullmatch(r' *\{([^\}]*)\} *', object)

        if match:
            fields = re.split(r', *', match.group(1))

            for field in fields:
                match = re.fullmatch(r' *([\_a-zA-Z0-9]+) *: *(.+)', field)

                if match:
                    name = match.group(1)
                    value = match.group(2)
                    yield (name, value)

                else:
                    raise RuntimeError("Object field must have syntax `name: value`")

        else:
            raise RuntimeError("Object must have syntax `{ ... }`")


    @classmethod
    def parse_optional_object(cls, object):
        if re.fullmatch(r' *', object) is None:
            yield from cls.parse_object(object)


    @staticmethod
    def parse_percent(value):
        match = re.fullmatch(r' *([\d\.]+)% *', value)

        if match:
            return { "percent": float(match.group(1)) * 0.01 }

        else:
            return int(value)


    @staticmethod
    def parse_bool(value):
        value = value.strip()

        if value == "true":
            return True

        elif value == "false":
            return False

        else:
            raise RuntimeError("Boolean must be true or false")


    @classmethod
    def parse_region(cls, region):
        x = 0
        y = 0
        width = { "percent": 1.0 }
        height = { "percent": 1.0 }
        strength = 1.0
        feather = 0
        isolated = False

        for (key, value) in cls.parse_optional_object(region):
            if key == "x":
                x = cls.parse_percent(value)
            elif key == "y":
                y = cls.parse_percent(value)
            elif key == "width":
                width = cls.parse_percent(value)
            elif key == "height":
                height = cls.parse_percent(value)
            elif key == "strength":
                strength = float(value)
            elif key == "feather":
                feather = cls.parse_percent(value)
            elif key == "isolated":
                isolated = cls.parse_bool(value)
            else:
                raise RuntimeError("Object field must be x, y, width, height, strength, feather, or isolated")

        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "strength": strength,
            "feather": feather,
            "isolated": isolated,
        }


    @classmethod
    def parse_mask_region(cls, region):
        match = re.fullmatch(r'([^\{\}]*)(.*)', region)

        if match:
            name = match.group(1).strip()
            strength = 1.0
            isolated = False

            if name == "":
                raise RuntimeError("MASK-REGION must have a name")

            for (key, value) in cls.parse_optional_object(match.group(2)):
                if key == "strength":
                    strength = float(value)
                elif key == "isolated":
                    isolated = cls.parse_bool(value)
                else:
                    raise RuntimeError("Object field must be strength or isolated")

            return {
                "name": name,
                "strength": strength,
                "isolated": isolated,
            }

        else:
            raise RuntimeError("Invalid syntax for MASK-REGION")


    @classmethod
    def parse_lines(cls, text, root, seen):
        output = []

        # Clean up the text so it doesn't have any tabs
        text = re.sub(r'\t+', r' ', text)

        bundles = []
        prompts = []
        region = None
        mask_region = None


        def process_break(reset_bundles):
            nonlocal bundles, prompts, region, mask_region

            if len(bundles) > 0:
                bundles[-1]["children"].extend(prompts)

                if reset_bundles:
                    output.append({ "bundles": bundles })
                    bundles = []

            elif len(prompts) > 0:
                output.append({ "chunk": prompts, "region": region, "mask-region": mask_region })

            prompts = []
            region = None
            mask_region = None


        def process_function(prompt, weight):
            function = re.fullmatch(r'<([a-z\-]+):([^>]*)>', prompt)

            if function:
                name = function.group(1)
                value = function.group(2).strip()

                if name == "bundle":
                    prompts.append({
                        "bundle": value,
                        "weight": weight,
                    })

                elif name == "bundle-inline":
                    prompts.append({
                        "bundle-inline": value,
                        "weight": weight,
                    })

                elif name == "lora":
                    prompts.append({
                        "lora": value,
                        "weight": weight,
                    })

                else:
                    raise RuntimeError("Unknown function {}".format(prompt))

            else:
                prompts.append({
                    "prompt": prompt,
                    "weight": weight,
                })


        def process_line(line):
            nonlocal region, mask_region

            if re.fullmatch(r'(?:BREAK|\-{3,})', line):
                process_break(True)
                return


            match = re.fullmatch(r'BUNDLE: *(.*)', line)

            if match:
                process_break(False)
                bundles.append({ "name": match.group(1), "children": [] })
                return


            match = re.fullmatch(r'FILE: *(.*)', line)

            if match:
                process_break(True)

                for path in cls.get_paths(match.group(1), root):
                    if not path in seen:
                        seen.add(path)

                        with open(path, "r", encoding="utf-8") as file:
                            output.extend(cls.parse_lines(file.read(), os.path.dirname(path), seen))
                return


            match = re.fullmatch(r'REGION: *(.*)', line)

            if match:
                process_break(True)
                region = cls.parse_region(match.group(1))
                return


            match = re.fullmatch(r'MASK-REGION: *(.*)', line)

            if match:
                process_break(True)
                mask_region = cls.parse_mask_region(match.group(1))
                return


            if re.search(r'BREAK', line) is not None:
                raise RuntimeError("Invalid BREAK found in text:\n\n" + text)

            if re.search(r'\-{3,}', line) is not None:
                raise RuntimeError("Invalid --- found in text:\n\n" + text)

            if re.search(r'BUNDLE:', line) is not None:
                raise RuntimeError("Invalid BUNDLE: found in text:\n\n" + text)

            if re.search(r'FILE:', line) is not None:
                raise RuntimeError("Invalid FILE: found in text:\n\n" + text)

            if re.search(r'REGION:', line) is not None:
                raise RuntimeError("Invalid REGION: found in text:\n\n" + text)

            if re.search(r'MASK-REGION:', line) is not None:
                raise RuntimeError("Invalid MASK-REGION: found in text:\n\n" + text)

            # Search for a weight for the line
            match = re.fullmatch(r'(.*)\* *([\-\d\.]+)', line)

            if match:
                prompt = match.group(1).strip()
                weight = float(match.group(2))

            else:
                prompt = line
                weight = 1.0

            # If there are multiple functions in a line, split them into separate prompts
            for prompt in re.split(r'(<[a-z\-]+:[^>]*>)[, ]*', prompt):
                if prompt != "":
                    process_function(prompt, weight)


        for line in text.splitlines():
            # TODO handle \\// and \\# properly

            # Remove // and # comments
            # They can be escaped by using \
            line = re.sub(r'(?<!\\)(?://|#).*', r'', line)

            # Remove the escaping \ before the comments
            line = re.sub(r'\\(?=//|#)', r'', line)

            line = line.strip()

            if line != "":
                process_line(line)

        process_break(True)
        return output


    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        # Copied from https://github.com/Comfy-Org/ComfyUI/blob/2687652530128435d5cdcfe2600751c8c4b75b88/folder_paths.py#L349-L366
        root = os.path.join(folder_paths.models_dir, "prompts")
        return io.NodeOutput(cls.parse_lines(text, root, set()))


class ParseYAML(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: ParseYAML",
            display_name="Parse YAML",
            category="prompt_helpers/prompt",
            description="Parses YAML text into JSON.",
            inputs=[
                io.String.Input("text", dynamic_prompts=False, multiline=True),
            ],
            outputs=[
                JSON.Output(display_name="JSON"),
            ],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        concat = []

        for json in yaml.safe_load_all(text):
            if not isinstance(json, list):
                raise RuntimeError("YAML is not an array.")

            concat.extend(json)

        return io.NodeOutput(concat)


class ApplyLoras(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: ApplyLoras",
            display_name="Apply Loras",
            description="Applies Loras from the JSON.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                JSON.Input("json", optional=True, default=[]),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
            ],
            enable_expand=True,
        )

    @staticmethod
    def lora_path(path):
        (_, ext) = os.path.splitext(path)

        if ext == "":
            path = path + ".safetensors"

        if folder_paths.get_full_path("loras", path):
            return path

        else:
            raise RuntimeError("Could not find lora: {}".format(path))

    @classmethod
    def process_loras(cls, json):
        seen = set()

        loras = []

        for item in json:
            for prompts in item["chunks"]:
                for item in prompts:
                    if "lora" in item:
                        path = cls.lora_path(item["lora"])
                        weight = item["weight"]

                        if path in seen:
                            raise RuntimeError("Duplicate lora: {}".format(path))
                        else:
                            seen.add(path)

                        if weight < 0.0:
                            raise RuntimeError("Loras must have a positive weight.")

                        loras.append({
                            "path": path,
                            "weight": weight,
                        })

        return loras

    @classmethod
    def execute(cls, model, clip, json) -> io.NodeOutput:
        graph = GraphBuilder()

        for lora in cls.process_loras(json):
            node = graph.node(
                "LoraLoader",
                model=model,
                clip=clip,
                lora_name=lora["path"],
                strength_model=lora["weight"],
                strength_clip=lora["weight"],
            )

            model = node.out(0)
            clip = node.out(1)

        return io.NodeOutput(model, clip, expand=graph.finalize())


class ConcatenateJson(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplatePrefix(input=JSON.Input("json"), prefix="json", min=1, max=20)

        return io.Schema(
            node_id="prompt_helpers: ConcatenateJson",
            display_name="Concatenate JSON",
            category="prompt_helpers/prompt",
            description="Concatenates two JSON into a single JSON.",
            inputs=[
                io.Autogrow.Input("jsons", template=template),
            ],
            outputs=[
                JSON.Output(display_name="JSON"),
            ],
        )

    @classmethod
    def execute(cls, jsons) -> io.NodeOutput:
        concat = []

        for key, json in jsons.items():
            if not isinstance(json, list):
                raise RuntimeError("{} is not an array.".format(key))

            concat.extend(json)

        return io.NodeOutput(concat)


class DebugJSONPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: DebugJSONPrompt",
            display_name="Debug JSON Prompt",
            category="prompt_helpers/prompt",
            description="Displays JSON prompt for debug purposes.",
            inputs=[
                io.Custom("JSON_PROMPT").Input("json_prompt"),
            ],
            outputs=[
                io.String.Output(display_name="CHUNKS", tooltip="List of strings, one string per chunk"),

                io.String.Output(display_name="MERGED", tooltip="Merges all of the chunks into a single string"),
            ],
        )

    @classmethod
    def execute(cls, json_prompt) -> io.NodeOutput:
        chunks = ProcessJson.serialize_chunks(json_prompt)

        return io.NodeOutput(
            dumps(chunks, indent=2),
            " ".join(chunks),
        )


class DebugJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: DebugJSON",
            display_name="Debug JSON",
            category="prompt_helpers/prompt",
            description="Displays JSON for debug purposes.",
            inputs=[
                JSON.Input("json"),
            ],
            outputs=[
                io.String.Output(display_name="ORIGINAL", tooltip="Original JSON input as a string."),
                io.String.Output(display_name="PROCESSED", tooltip="Processed JSON."),

                io.Custom("JSON_PROMPT").Output(display_name="POSITIVE", tooltip="JSON that only contains positive prompts."),
                io.Custom("JSON_PROMPT").Output(display_name="NEGATIVE", tooltip="JSON that only contains negative prompts."),

                io.String.Output(display_name="LORAS", tooltip="List of lora paths and weights"),
            ],
        )

    @classmethod
    def execute(cls, json) -> io.NodeOutput:
        processed = ProcessJson.process_json(json)

        positive = ProcessJson.only_positive(processed)
        negative = ProcessJson.only_negative(processed)

        loras = ApplyLoras.process_loras(processed)

        return io.NodeOutput(
            dumps(json, indent=2),
            dumps(processed, indent=2),
            positive,
            negative,
            dumps(loras, indent=2),
        )


class PromptToggle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: PromptToggle",
            display_name="Prompt Toggle",
            category="prompt_helpers/prompt",
            description="Makes it easy to toggle prompts on and off.",
            inputs=[
                io.String.Input("text", multiline=True, dynamic_prompts=False),
            ],
            outputs=[
                JSON.Output(display_name="JSON"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        graph = GraphBuilder()

        parse_lines = graph.node("prompt_helpers: ParseLines", text=text)

        return io.NodeOutput(parse_lines.out(0), expand=graph.finalize())
