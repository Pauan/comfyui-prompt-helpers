from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import re
import torch
import yaml
from json import (dumps)


class ProcessJson:
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
    def process_children(cls, bundles, seen, outer_weight, children):
        output = []

        for item in children:
            if "prompt" in item:
                if item.get("enabled", True):
                    output.append({
                        "prompt": item["prompt"],
                        "weight": outer_weight * item.get("weight", 1.0),
                    })

            elif "bundle" in item:
                if item.get("enabled", True):
                    name = item["bundle"]

                    if not name in bundles:
                        raise RuntimeError("ERROR: Bundle {} not found.".format(name))

                    if name in seen:
                        raise RuntimeError("ERROR: Infinite recursion when inserting bundle {}".format(name))

                    bundle = bundles[name]

                    output.extend(cls.process_children(
                        bundles,
                        seen.union({ name }),
                        outer_weight * item.get("weight", 1.0),
                        bundle["children"],
                    ))

            else:
                raise RuntimeError("ERROR: Must be prompt or bundle.")

        return output


    """
        Expands bundles, normalizing all weights.
    """
    @classmethod
    def process_json(cls, json):
        if not isinstance(json, list):
            raise RuntimeError("ERROR: JSON is not an array.")

        bundles = {}
        seen = frozenset()

        output = []

        for item in json:
            if "bundles" in item:
                if not isinstance(item["bundles"], list):
                    raise RuntimeError("ERROR: Bundles is not an array.")

                for bundle in item["bundles"]:
                    if bundle.get("enabled", True):
                        name = bundle["name"]

                        if name in bundles:
                            raise RuntimeError("ERROR: Duplicate bundle {}".format(name))

                        else:
                            bundles[name] = bundle

            elif "chunk" in item:
                if not isinstance(item["chunk"], list):
                    raise RuntimeError("ERROR: Chunk is not an array.")

                prompts = cls.process_children(bundles, seen, 1.0, item["chunk"])

                if len(prompts) > 0:
                    output.append({
                        "chunk": prompts
                    })

            else:
                raise RuntimeError("ERROR: Root must be either bundles or chunk.")

        return output


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
                        prompt = cls.cleanup_prompt(item["prompt"])

                        if prompt != "":
                            prompt = prompt + ","
                            weight = item["weight"]

                            if weight == 1.0:
                                prompts.append(prompt)
                            else:
                                prompts.append("({}:{}),".format(prompt, weight))

                if len(prompts) > 0:
                    chunks.append(" ".join(prompts))

        return chunks


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
                io.Custom("JSON").Output(),
            ],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        chunks = []

        # Clean up the text so it doesn't have any tabs
        text = re.sub(r'\t+', r' ', text)

        # Split the text into chunks for each BREAK
        for chunk in re.split(r'(?:^|(?<=[\n\r])) *BREAK *(?:$|[\n\r]+)', text):
            if chunk != "":
                if re.match(r'BREAK', chunk) is not None:
                    raise RuntimeError("Invalid BREAK found in text:\n\n" + text)

                prompts = []

                # TODO handle \\// and \\# properly
                for line in chunk.splitlines():
                    # Remove // and # comments
                    # They can be escaped by using \
                    line = re.sub(r'(?<!\\)(?://|#).*', r'', line)

                    # Remove the escaping \ before the comments
                    line = re.sub(r'\\(?=//|#)', r'', line)

                    line = line.strip()

                    if line != "":
                        # Search for a weight for the line
                        match = re.search(r'(.*)\* *([\d\.]+)$', line)

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
                                    prompts.append({
                                        "prompt": prompt,
                                        "weight": weight,
                                    })

                # If the chunk is not empty
                if len(prompts) > 0:
                    chunks.append(prompts)

        return io.NodeOutput([{ "chunk": chunk } for chunk in chunks])


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
                io.Custom("JSON").Output(),
            ],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        concat = []

        for json in yaml.safe_load_all(text):
            if not isinstance(json, list):
                raise RuntimeError("ERROR: YAML is not an array.")

            concat.extend(json)

        return io.NodeOutput(concat)


class ConcatenateJson(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplatePrefix(input=io.Custom("JSON").Input("json"), prefix="json", min=1, max=20)

        return io.Schema(
            node_id="prompt_helpers: ConcatenateJson",
            display_name="Concatenate JSON",
            category="prompt_helpers/prompt",
            description="Concatenates two JSON into a single JSON.",
            inputs=[
                io.Autogrow.Input("jsons", template=template),
            ],
            outputs=[
                io.Custom("JSON").Output(),
            ],
        )

    @classmethod
    def execute(cls, jsons) -> io.NodeOutput:
        concat = []

        for key, json in jsons.items():
            if not isinstance(json, list):
                raise RuntimeError("ERROR: {} is not an array.".format(key))

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
                io.Custom("JSON").Input("json"),
            ],
            outputs=[
                io.String.Output(display_name="ORIGINAL", tooltip="Original JSON input."),

                io.String.Output(display_name="PROCESSED", tooltip="JSON after it has been processed and expanded."),

                # TODO replace this when preview supports output lists
                io.Custom("LIST").Output(display_name="CHUNKS", tooltip="List of strings, one string per chunk"),
                #io.String.Output(is_output_list=True, display_name="POSITIVE", tooltip="List of strings, one string per chunk"),
            ],
        )

    @classmethod
    def execute(cls, json) -> io.NodeOutput:
        processed = ProcessJson.process_json(json)

        chunks = ProcessJson.serialize_chunks(processed)

        return io.NodeOutput(
            dumps(json, indent=2),
            dumps(processed, indent=2),
            chunks,
        )


class FromJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: FromJSON",
            display_name="From JSON",
            category="prompt_helpers/prompt",
            description="Converts a JSON prompt into a conditioning.",
            inputs=[
                io.Clip.Input("clip"),
                io.Custom("JSON").Input("json"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, json) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        json = ProcessJson.process_json(json)

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
            return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

        else:
            # Concatenate the tensors
            concatenated = torch.cat([chunk[0] for chunk in chunks], 1)

            # This always returns the metadata for the first chunk, this matches the behavior of ConditioningConcat
            metadata = chunks[0][1].copy()

            return io.NodeOutput([[concatenated, metadata]])


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
                io.Custom("JSON").Output(),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        graph = GraphBuilder()

        parse_lines = graph.node("prompt_helpers: ParseLines", text=text)

        return io.NodeOutput(parse_lines.out(0), expand=graph.finalize())
