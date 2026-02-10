from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import os.path
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


    @classmethod
    def process_children(cls, bundles, seen_bundles, seen_prompts, outer_weight, children):
        output = []

        for item in children:
            if "prompt" in item:
                if item.get("enabled", True):
                    prompt = cls.cleanup_prompt(item["prompt"])

                    if prompt != "":
                        if prompt in seen_prompts:
                            raise RuntimeError("Duplicate prompt: {}".format(prompt))
                        else:
                            seen_prompts.add(prompt)

                        output.append({
                            "prompt": prompt,
                            "weight": round(outer_weight * item.get("weight", 1.0), 2),
                        })

            elif "lora" in item:
                if item.get("enabled", True):
                    output.append({
                        "lora": item["lora"],
                        "weight": round(outer_weight * item.get("weight", 1.0), 2),
                    })

            elif "bundle" in item:
                if item.get("enabled", True):
                    name = item["bundle"]

                    if not name in bundles:
                        raise RuntimeError("Bundle {} not found.".format(name))

                    if name in seen_bundles:
                        raise RuntimeError("Infinite recursion when inserting bundle {}".format(name))

                    bundle = bundles[name]

                    output.extend(cls.process_children(
                        bundles,
                        seen_bundles.union({ name }),
                        seen_prompts,
                        outer_weight * item.get("weight", 1.0),
                        bundle["children"],
                    ))

            else:
                raise RuntimeError("Unknown type.")

        return output


    """
        Expands bundles, normalizing all weights.
    """
    @classmethod
    def process_json(cls, json):
        if not isinstance(json, list):
            raise RuntimeError("JSON is not an array.")

        bundles = {}
        seen_bundles = frozenset()
        seen_prompts = set()

        output = []

        for item in json:
            if "bundles" in item:
                if not isinstance(item["bundles"], list):
                    raise RuntimeError("Bundles is not an array.")

                for bundle in item["bundles"]:
                    if bundle.get("enabled", True):
                        name = bundle["name"]

                        if name in bundles:
                            raise RuntimeError("Duplicate bundle: {}".format(name))

                        else:
                            bundles[name] = bundle

            elif "chunk" in item:
                if not isinstance(item["chunk"], list):
                    raise RuntimeError("Chunk is not an array.")

                prompts = cls.process_children(bundles, seen_bundles, seen_prompts, 1.0, item["chunk"])

                if len(prompts) > 0:
                    output.append({
                        "chunk": prompts
                    })

            else:
                raise RuntimeError("Root must be either bundles or chunk.")

        return output


    """
        Filters JSON to only have positive weights.
    """
    @classmethod
    def only_positive(cls, json):
        chunks = []

        for item in json:
            if "chunk" in item:
                prompts = []

                for item in item["chunk"]:
                    if "prompt" in item:
                        if item["weight"] > 0.0:
                            prompts.append(item)

                if len(prompts) > 0:
                    chunks.append({
                        "chunk": prompts
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

                for item in item["chunk"]:
                    if "prompt" in item:
                        if item["weight"] < 0.0:
                            item = item.copy()
                            item["weight"] = -item["weight"]
                            prompts.append(item)

                if len(prompts) > 0:
                    chunks.append({
                        "chunk": prompts
                    })

        return chunks


    """
        Converts chunks into text strings, ready to be CLIP encoded.
    """
    @classmethod
    def serialize_chunks(cls, json):
        chunks = []

        for item in json:
            if "chunk" in item:
                prompts = []

                for item in item["chunk"]:
                    if "prompt" in item:
                        prompt = item["prompt"] + ","
                        weight = item["weight"]

                        if weight == 1.0:
                            prompts.append(prompt)
                        else:
                            prompts.append("({}:{}),".format(prompt, weight))

                if len(prompts) > 0:
                    chunks.append(" ".join(prompts))

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
    def execute(cls, text) -> io.NodeOutput:
        output = []

        # Clean up the text so it doesn't have any tabs
        text = re.sub(r'\t+', r' ', text)

        bundles = []
        prompts = []


        def process_break(reset_bundles):
            nonlocal bundles, prompts

            if len(bundles) > 0:
                bundles[-1]["children"].extend(prompts)

                if reset_bundles:
                    output.append({ "bundles": bundles })
                    bundles = []

            elif len(prompts) > 0:
                output.append({ "chunk": prompts })

            prompts = []


        def process_line(line):
            if re.match(r'BREAK', line) is not None:
                raise RuntimeError("Invalid BREAK found in text:\n\n" + text)

            if re.match(r'\-{3,}', line) is not None:
                raise RuntimeError("Invalid --- found in text:\n\n" + text)

            if re.match(r'BUNDLE:', line) is not None:
                raise RuntimeError("Invalid BUNDLE: found in text:\n\n" + text)

            # TODO handle \\// and \\# properly

            # Remove // and # comments
            # They can be escaped by using \
            line = re.sub(r'(?<!\\)(?://|#).*', r'', line)

            # Remove the escaping \ before the comments
            line = re.sub(r'\\(?=//|#)', r'', line)

            line = line.strip()

            if line != "":
                # Search for a weight for the line
                match = re.search(r'(.*)\* *([\-\d\.]+)$', line)

                if match:
                    prompt = match.group(1)
                    weight = float(match.group(2))

                else:
                    prompt = line
                    weight = 1.0

                # If there are multiple bundles in a line, split them into separate prompts
                for prompt in re.split(r'(<[a-z]+:[^>]*>)[, ]*', prompt):
                    if prompt != "":
                        bundle = re.fullmatch(r'<bundle:([^>]*)>', prompt)

                        if bundle:
                            prompts.append({
                                "bundle": bundle.group(1).strip(),
                                "weight": weight,
                            })

                        else:
                            lora = re.fullmatch(r'<lora:([^>]*)>', prompt)

                            if lora:
                                prompts.append({
                                    "lora": lora.group(1).strip(),
                                    "weight": weight,
                                })

                            else:
                                prompts.append({
                                    "prompt": prompt,
                                    "weight": weight,
                                })


        for line in text.splitlines():
            if line != "":
                if re.fullmatch(r' *(?:BREAK|\-{3,}) *', line):
                    process_break(True)

                else:
                    match = re.fullmatch(r' *BUNDLE:(.*)', line)

                    if match:
                        process_break(False)
                        bundles.append({ "name": match.group(1).strip(), "children": [] })

                    else:
                        process_line(line)

        process_break(True)

        return io.NodeOutput(output)


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
    def execute(cls, model, clip, json) -> io.NodeOutput:
        graph = GraphBuilder()

        seen = set()

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

                        node = graph.node(
                            "LoraLoader",
                            model=model,
                            clip=clip,
                            lora_name=path,
                            strength_model=weight,
                            strength_clip=weight,
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
                io.String.Output(display_name="ORIGINAL", tooltip="Original JSON input."),

                io.String.Output(display_name="POSITIVE", tooltip="Processed JSON that only contains positive prompts."),
                io.String.Output(display_name="NEGATIVE", tooltip="Processed JSON that only contains negative prompts."),

                # TODO replace this when preview supports output lists
                io.Custom("LIST").Output(display_name="POSITIVE_CHUNKS", tooltip="List of strings, one string per positive chunk"),
                io.Custom("LIST").Output(display_name="NEGATIVE_CHUNKS", tooltip="List of strings, one string per negative chunk"),
                #io.String.Output(is_output_list=True, display_name="POSITIVE", tooltip="List of strings, one string per chunk"),
            ],
        )

    @classmethod
    def execute(cls, json) -> io.NodeOutput:
        processed = ProcessJson.process_json(json)

        positive = ProcessJson.only_positive(processed)
        negative = ProcessJson.only_negative(processed)

        positive_chunks = ProcessJson.serialize_chunks(positive)
        negative_chunks = ProcessJson.serialize_chunks(negative)

        return io.NodeOutput(
            dumps(json, indent=2),
            dumps(positive, indent=2),
            dumps(negative, indent=2),
            positive_chunks,
            negative_chunks,
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
        )

    @staticmethod
    def encode(clip, json):
        chunks_text = ProcessJson.serialize_chunks(json)

        chunks = []

        for chunk in chunks_text:
            # Encode using the same logic as CLIPTextEncode
            tokens = clip.tokenize(chunk)

            encoded = clip.encode_from_tokens_scheduled(tokens)
            assert len(encoded) == 1

            encoded = encoded[0]
            assert len(encoded) == 2

            chunks.append(encoded)

        # Return an empty conditioning
        if len(chunks) == 0:
            tokens = clip.tokenize("")
            return clip.encode_from_tokens_scheduled(tokens)

        else:
            # Concatenate the tensors
            concatenated = torch.cat([chunk[0] for chunk in chunks], 1)

            # This always returns the metadata for the first chunk, this matches the behavior of ConditioningConcat
            metadata = chunks[0][1].copy()

            return [[concatenated, metadata]]

    @classmethod
    def execute(cls, clip, json) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        positive = ProcessJson.only_positive(json)
        negative = ProcessJson.only_negative(json)

        return io.NodeOutput(cls.encode(clip, positive), cls.encode(clip, negative))


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
