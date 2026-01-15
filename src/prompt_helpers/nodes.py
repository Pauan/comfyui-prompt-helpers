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

        # Split the text into chunks for each BREAK
        for chunk in re.split(r'(?:^|(?<=[\n\r]))[ \t]*BREAK[ \t]*(?:$|[\n\r]+)', text):
            if chunk != "":
                if re.match(r'BREAK', chunk) is not None:
                    raise RuntimeError("Invalid BREAK found in text:\n\n" + text)

                # Clean up the text so it doesn't have any newlines
                chunk = re.sub(r'(?:[\n\r]+|$)', r', ', chunk)

                # Clean up the text so it doesn't have repeated periods or commas
                chunk = re.sub(r'([\.,])[\., \t]+', r'\1 ', chunk)

                # Clean up the text so it removes periods, commas, or spaces before the weight
                chunk = re.sub(r'[\., \t]+(:[\d\.]+\))', r'\1', chunk)

                # Clean up the text so it removes periods, commas, or spaces at the start
                chunk = re.sub(r'^[\., \t]+', r'', chunk)

                # Clean up the text so it removes spaces at the end
                chunk = re.sub(r'[ \t]+$', r'', chunk)

                # Clean up the text so it doesn't have repeated spaces
                chunk = re.sub(r' {2,}', r' ', chunk)

                # If the chunk is not empty
                if re.fullmatch(r'[\., \t]*', chunk) is None:
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
