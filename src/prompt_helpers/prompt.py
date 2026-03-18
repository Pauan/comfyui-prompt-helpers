from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import os.path
import functools
import re
import torch
import yaml
import folder_paths
from json import (dumps)


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


class Chunk:
    def __init__(self, region):
        self.prompts = []
        self.region = region
        self.pushed = False


class ProcessState:
    def __init__(self):
        self.seen_prompts = set()
        self.bundles = {}
        self.chunks = []


    @staticmethod
    def cleanup_prompt(prompt):
        # Replace tabs with a space
        prompt = re.sub(r'\t+', r' ', prompt)

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
        prompt = re.sub(r'(?<!\\)\(', r'\\(', prompt)

        # Replaces ) with \\)
        prompt = re.sub(r'(?<!\\)\)', r'\\)', prompt)

        return prompt


    def push_chunk(self, chunk):
        if not chunk.pushed:
            assert len(chunk.prompts) > 0
            chunk.pushed = True
            self.chunks.append({ "chunk": chunk.prompts, "region": chunk.region })


    def process_bundle(self, seen_bundles, outer_weight, item, name, chunk):
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
            )


    def process_children(self, seen_bundles, outer_weight, children, chunk):
        for item in children:
            if "prompt" in item:
                if item.get("enabled", True):
                    prompt = ProcessState.cleanup_prompt(item["prompt"])

                    if prompt != "":
                        key = (id(chunk.region), prompt)

                        if key in self.seen_prompts:
                            raise RuntimeError("Duplicate prompt: {}".format(prompt))
                        else:
                            self.seen_prompts.add(key)

                        chunk.prompts.append({
                            "prompt": prompt,
                            "weight": round(outer_weight * item.get("weight", 1.0), 2),
                        })

                        self.push_chunk(chunk)

            elif "lora" in item:
                if item.get("enabled", True):
                    chunk.prompts.append({
                        "lora": item["lora"],
                        "weight": round(outer_weight * item.get("weight", 1.0), 2),
                    })

                    self.push_chunk(chunk)

            elif "bundle" in item:
                self.process_bundle(seen_bundles, outer_weight, item, item["bundle"], Chunk(None))

            elif "bundle-inline" in item:
                self.process_bundle(seen_bundles, outer_weight, item, item["bundle-inline"], chunk)

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

                self.process_children(frozenset(), 1.0, item["chunk"], Chunk(item.get("region", None)))

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


    """
        Filters JSON to only have positive weights.
    """
    @classmethod
    def only_positive(cls, json):
        chunks = []

        for item in json:
            if "chunk" in item:
                prompts = []
                region = item.get("region", None)

                for item in item["chunk"]:
                    if "prompt" in item:
                        if item["weight"] > 0.0:
                            prompts.append(item)

                if len(prompts) > 0:
                    chunks.append({
                        "chunk": prompts,
                        "region": region,
                    })

        return chunks


    """
        Filters JSON to only have negative weights.
    """
    @classmethod
    def only_negative(cls, json):
        chunks = []

        for item in json:
            if "chunk" in item:
                prompts = []
                region = item.get("region", None)

                for item in item["chunk"]:
                    if "prompt" in item:
                        if item["weight"] < 0.0:
                            item = item.copy()
                            item["weight"] = -item["weight"]
                            prompts.append(item)

                if len(prompts) > 0:
                    chunks.append({
                        "chunk": prompts,
                        "region": region,
                    })

        return chunks


    """
        Converts chunk into text strings, ready to be CLIP encoded.
    """
    @staticmethod
    def serialize_chunk(children):
        prompts = []

        for item in children:
            if "prompt" in item:
                prompt = item["prompt"] + ","
                weight = item["weight"]

                if weight == 1.0:
                    prompts.append(prompt)
                else:
                    prompts.append("({}:{}),".format(prompt, weight))

        if len(prompts) > 0:
            return " ".join(prompts)

    """
        Converts chunks into text strings, ready to be CLIP encoded.
    """
    @classmethod
    def serialize_chunks(cls, json):
        chunks = []

        for item in json:
            if "chunk" in item:
                prompt = cls.serialize_chunk(item["chunk"])

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
    def read_file(path):
        # Copied from https://github.com/Comfy-Org/ComfyUI/blob/2687652530128435d5cdcfe2600751c8c4b75b88/folder_paths.py#L349-L366
        full_path = os.path.join(os.path.join(folder_paths.models_dir, "prompts"), os.path.relpath(os.path.join("/", path), "/"))

        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            raise RuntimeError("Could not find file: {}".format(path))


    @staticmethod
    def parse_region(region):
        match = re.fullmatch(r' *\{([^\}]*)\} *', region)

        if match:
            fields = re.split(r', *', match.group(1))

            x = None
            y = None
            width = None
            height = None
            strength = 1.0

            for field in fields:
                match = re.fullmatch(r' *([a-z]+) *: *([^ ]+) *', field)

                if match:
                    name = match.group(1)
                    value = match.group(2)

                    if name == "x":
                        x = float(value)
                    elif name == "y":
                        y = float(value)
                    elif name == "width":
                        width = float(value)
                    elif name == "height":
                        height = float(value)
                    elif name == "strength":
                        strength = float(value)
                    else:
                        raise RuntimeError("Region field must be x, y, width, height, or strength")

                else:
                    raise RuntimeError("Region field must have syntax `name: value`")

            if x is None:
                raise RuntimeError("Region is missing x")

            if y is None:
                raise RuntimeError("Region is missing y")

            if width is None:
                raise RuntimeError("Region is missing width")

            if height is None:
                raise RuntimeError("Region is missing height")

            return {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "strength": strength,
            }

        else:
            raise RuntimeError("Region must have syntax `{ ... }`")


    @classmethod
    def parse_lines(cls, text):
        output = []

        # Clean up the text so it doesn't have any tabs
        text = re.sub(r'\t+', r' ', text)

        bundles = []
        prompts = []
        region = None


        def process_break(reset_bundles):
            nonlocal bundles, prompts, region

            if len(bundles) > 0:
                bundles[-1]["children"].extend(prompts)

                if reset_bundles:
                    output.append({ "bundles": bundles })
                    bundles = []

            elif len(prompts) > 0:
                output.append({ "chunk": prompts, "region": region })

            prompts = []
            region = None


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
            nonlocal region

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
                output.extend(cls.parse_lines(cls.read_file(match.group(1))))
                return


            match = re.fullmatch(r'REGION: *(.*)', line)

            if match:
                process_break(True)
                region = cls.parse_region(match.group(1))
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
        return io.NodeOutput(cls.parse_lines(text))


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
            if "chunk" in item:
                for item in item["chunk"]:
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
            positive,
            negative,
            dumps(loras, indent=2),
        )


class FromJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: FromJSON",
            display_name="From JSON",
            description="Converts a JSON prompt into a positive and negative conditioning.",
            inputs=[
                io.Clip.Input("clip"),
                JSON.Input("json"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="POSITIVE"),
                io.Conditioning.Output(display_name="NEGATIVE"),
            ],
            enable_expand=True,
        )

    @staticmethod
    def encode(graph, clip, json):
        global_chunks = []
        region_chunks = []

        for item in json:
            if "chunk" in item:
                prompt = ProcessJson.serialize_chunk(item["chunk"])

                if prompt is not None:
                    region = item.get("region", None)

                    chunk = graph.node("CLIPTextEncode", text=prompt, clip=clip).out(0)

                    if region is None:
                        global_chunks.append(chunk)
                    else:
                        region_chunks.append((chunk, region))


        if len(global_chunks) == 0:
            global_prompt = None

            # There must always be at least one chunk, so we create an empty chunk
            final_chunks = [
                graph.node("CLIPTextEncode", text="", clip=clip).out(0)
            ]

        # Uses ConditioningConcat to combine the global chunks
        else:
            global_prompt = functools.reduce(lambda x, y: graph.node("ConditioningConcat", conditioning_to=x, conditioning_from=y).out(0), global_chunks)

            final_chunks = [
                global_prompt
            ]


        for (chunk, region) in region_chunks:
            # Concats the global prompt with the region prompt
            # If we don't do this then the global prompt will have a weak effect inside the region and it causes artifacting
            if global_prompt is not None:
                chunk = graph.node("ConditioningConcat", conditioning_to=global_prompt, conditioning_from=chunk).out(0)

            chunk = graph.node(
                "ConditioningSetAreaPercentage",
                conditioning=chunk,
                x=region["x"],
                y=region["y"],
                width=region["width"],
                height=region["height"],
                strength=region.get("strength", 1.0),
            ).out(0)

            final_chunks.append(chunk)


        assert len(final_chunks) > 0

        # Combines the regions together
        # We have to use ConditioningCombine because ConditioningConcat does not work with ConditioningSetAreaPercentage
        return functools.reduce(lambda x, y: graph.node("ConditioningCombine", conditioning_1=x, conditioning_2=y).out(0), final_chunks)


    @classmethod
    def execute(cls, clip, json) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        graph = GraphBuilder()

        positive = ProcessJson.only_positive(json)
        negative = ProcessJson.only_negative(json)

        return io.NodeOutput(
            cls.encode(graph, clip, positive),
            cls.encode(graph, clip, negative),
            expand=graph.finalize(),
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
