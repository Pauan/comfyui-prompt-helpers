from inspect import cleandoc
from comfy.comfy_types import IO
from comfy_execution.graph_utils import GraphBuilder
import comfy
import datetime
import re
import torch


# Copied from Comfy-Org/ComfyUI/nodes.py
MAX_RESOLUTION=16384


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
    CATEGORY = "prompt_helpers"

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


class EZBlank:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the images in pixels."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of the images in pixels."}),
            },
        }

    RETURN_TYPES = ("IMAGE_SETTINGS",)
    FUNCTION = "settings"
    CATEGORY = "prompt_helpers/image"

    def settings(self, width, height):
        return ({
            "type": "BLANK",
            "width": width,
            "height": height,
        },)


class EZImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "How much the image should influence the result. Higher number means it closely matches the image, lower number means more random."}),
            },
        }

    RETURN_TYPES = ("IMAGE_SETTINGS",)
    FUNCTION = "settings"
    CATEGORY = "prompt_helpers/image"

    def settings(self, image, image_weight):
        return ({
            "type": "IMAGE",
            "image": image,
            "image_weight": image_weight,
        },)


class EZInpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "image_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "How much the image should influence the result. Higher number means it closely matches the image, lower number means more random."}),
                "grow_mask": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE_SETTINGS",)
    FUNCTION = "settings"
    CATEGORY = "prompt_helpers/image"

    def settings(self, image, mask, image_weight, grow_mask):
        return ({
            "type": "INPAINT",
            "image": image,
            "mask": mask,
            "image_weight": image_weight,
            "grow_mask": grow_mask,
        },)


class EZSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_sde", "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            },
        }

    RETURN_TYPES = ("SAMPLER_SETTINGS",)
    FUNCTION = "settings"
    CATEGORY = "prompt_helpers/sampler"

    def settings(self, seed, steps, cfg, sampler_name, scheduler):
        return ({
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        },)


class EZTimestamp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "folder": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, image, folder):
        filename = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")

        # TODO make this work on Windows too
        prefix = folder + "/" + filename

        return (prefix,)


class EZGenerate:
    """

    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent.", "rawLink": True}),
                "vae": ("VAE", {"rawLink": True}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image.", "rawLink": True}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image.", "rawLink": True}),

                "folder": ("STRING", {"default": "Random", "tooltip": "The folder that the images will be saved in."}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "rawLink": True}),

                "image": ("IMAGE_SETTINGS",),
                "sampler": ("SAMPLER_SETTINGS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "generate"
    CATEGORY = "prompt_helpers"


    def generate_text(self, folder, image, sampler, **kwargs):
        graph = GraphBuilder()

        empty_image = graph.node("EmptyLatentImage", width=image["width"], height=image["height"], batch_size=kwargs["batch_size"])

        ksampler = graph.node(
            "KSampler",
            model=kwargs["model"],
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=sampler["cfg"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=kwargs["positive"],
            negative=kwargs["negative"],
            latent_image=empty_image.out(0),
            denoise=1.0,
        )

        vae_decode = graph.node(
            "VAEDecodeTiled",
            samples=ksampler.out(0),
            vae=kwargs["vae"],
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )

        timestamp = graph.node("prompt_helpers: EZTimestamp", image=vae_decode.out(0), folder=folder)

        return {
            "result": (vae_decode.out(0), timestamp.out(0)),
            "expand": graph.finalize(),
        }


    def generate_image(self, folder, image, sampler, **kwargs):
        graph = GraphBuilder()

        vae_encode = graph.node("VAEEncode", pixels=image["image"], vae=kwargs["vae"])

        repeat_latent_batch = graph.node("RepeatLatentBatch", samples=vae_encode.out(0), amount=kwargs["batch_size"])

        ksampler = graph.node(
            "KSampler",
            model=kwargs["model"],
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=sampler["cfg"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=kwargs["positive"],
            negative=kwargs["negative"],
            latent_image=repeat_latent_batch.out(0),
            denoise=(1.0 - image["image_weight"]),
        )

        vae_decode = graph.node(
            "VAEDecodeTiled",
            samples=ksampler.out(0),
            vae=kwargs["vae"],
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )

        timestamp = graph.node("prompt_helpers: EZTimestamp", image=vae_decode.out(0), folder=folder)

        return {
            "result": (vae_decode.out(0), timestamp.out(0)),
            "expand": graph.finalize(),
        }


    def generate_inpainting(self, folder, image, sampler, **kwargs):
        graph = GraphBuilder()

        grow_mask = graph.node("GrowMask", mask=image["mask"], expand=image["grow_mask"], tapered_corners=True)

        # VAEEncodeForInpaint doesn't support image_weight, so we use InpaintModelConditioning instead
        inpaint_model_conditioning = graph.node(
            "InpaintModelConditioning",
            positive=kwargs["positive"],
            negative=kwargs["negative"],
            vae=kwargs["vae"],
            pixels=image["image"],
            mask=grow_mask.out(0),
            noise_mask=True,
        )

        repeat_latent_batch = graph.node("RepeatLatentBatch", samples=inpaint_model_conditioning.out(2), amount=kwargs["batch_size"])

        ksampler = graph.node(
            "KSampler",
            model=kwargs["model"],
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=sampler["cfg"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=inpaint_model_conditioning.out(0),
            negative=inpaint_model_conditioning.out(1),
            latent_image=repeat_latent_batch.out(0),
            denoise=(1.0 - image["image_weight"]),
        )

        vae_decode = graph.node(
            "VAEDecodeTiled",
            samples=ksampler.out(0),
            vae=kwargs["vae"],
            tile_size=512,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )

        # ComfyUI changes the image even outside of the mask, so we overwrite the image
        # to guarantee that *only* the masked area will be changed
        repeat_image_batch = graph.node(
            "RepeatImageBatch",
            image=image["image"],
            amount=kwargs["batch_size"],
        )

        image_composite_masked = graph.node(
            "ImageCompositeMasked",
            destination=repeat_image_batch.out(0),
            source=vae_decode.out(0),
            mask=grow_mask.out(0),
            x=0,
            y=0,
            resize_source=False,
        )

        timestamp = graph.node("prompt_helpers: EZTimestamp", image=image_composite_masked.out(0), folder=folder)

        return {
            "result": (image_composite_masked.out(0), timestamp.out(0)),
            "expand": graph.finalize(),
        }


    def generate(self, folder, image, sampler, **kwargs):
        if image["type"] == "BLANK":
            return self.generate_text(folder=folder, image=image, sampler=sampler, **kwargs)

        elif image["type"] == "IMAGE":
            return self.generate_image(folder=folder, image=image, sampler=sampler, **kwargs)

        elif image["type"] == "INPAINT":
            return self.generate_inpainting(folder=folder, image=image, sampler=sampler, **kwargs)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "prompt_helpers: PromptToggle": PromptToggle,
    "prompt_helpers: EZGenerate": EZGenerate,
    "prompt_helpers: EZBlank": EZBlank,
    "prompt_helpers: EZImage": EZImage,
    "prompt_helpers: EZInpaint": EZInpaint,
    "prompt_helpers: EZSampler": EZSampler,
    "prompt_helpers: EZTimestamp": EZTimestamp,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "prompt_helpers: PromptToggle": "Prompt Toggle",
    "prompt_helpers: EZGenerate": "EZ Generate",
    "prompt_helpers: EZBlank": "EZ txt2img",
    "prompt_helpers: EZImage": "EZ img2img",
    "prompt_helpers: EZInpaint": "EZ Inpaint",
    "prompt_helpers: EZSampler": "EZ Sampler",
    "prompt_helpers: EZTimestamp": "EZ Timestamp",
}
