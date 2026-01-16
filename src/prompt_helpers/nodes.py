from inspect import cleandoc
from comfy.comfy_types import IO
import re
import torch


class PromptToggle:
    """

    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The prompt text."}),
            },
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "encode"
    CATEGORY = "conditioning/helpers"

    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        chunks = []

        # Clean up the text so it doesn't have any tabs
        text = re.sub(r'\t+', r' ', text)

        # Split the text into chunks for each BREAK
        for chunk in re.split(r'(?:^|(?<=[\n\r])) *BREAK *(?:$|[\n\r]+)', text):
            if chunk != "":
                if re.match(r'BREAK', chunk) is not None:
                    raise RuntimeError("Invalid BREAK found in text:\n\n" + text)

                # Clean up the text so it doesn't have any newlines
                chunk = re.sub(r'(?:[\n\r]+|$)', r', ', chunk)

                # Removes spaces before a period or comma
                chunk = re.sub(r' +(?=[\.,])', r'', chunk)

                # Removes repeated periods or commas
                chunk = re.sub(r'([\.,])[\., ]+', r'\1 ', chunk)

                # Adds a space after commas
                chunk = re.sub(r',(?![ \:]|$)', r', ', chunk)

                # Adds a space after periods
                chunk = re.sub(r'\.(?![ \:\d]|$)', r'. ', chunk)

                # Removes spaces around the weight
                chunk = re.sub(r' *: *([\d\.]+) *\) *', r':\1)', chunk)

                # Removes periods, commas, or spaces at the start
                chunk = re.sub(r'^[\., ]+', r'', chunk)

                # Removes spaces at the end
                chunk = re.sub(r' +$', r'', chunk)

                # Removes repeated spaces
                chunk = re.sub(r' {2,}', r' ', chunk)

                # If the chunk is not empty
                if re.fullmatch(r'[\., ]*', chunk) is None:
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
            return (clip.encode_from_tokens_scheduled(tokens),)

        else:
            # Concatenate the tensors
            concatenated = torch.cat([chunk[0] for chunk in chunks], 1)

            # This always returns the metadata for the first chunk, this matches the behavior of ConditioningConcat
            metadata = chunks[0][1].copy()

            return ([[concatenated, metadata]],)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "prompt_helpers: PromptToggle": PromptToggle
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "prompt_helpers: PromptToggle": "Prompt Toggle"
}
