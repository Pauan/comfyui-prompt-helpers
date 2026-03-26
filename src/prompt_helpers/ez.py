from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import comfy
import folder_paths
import datetime
import desktop_notifier
from .prompt import (JSON, ProcessJson)
from .upscale import get_image_tiles
from .image import (Crop, Detail, ProcessImage)
from .utils import (fold)


# Copied from Comfy-Org/ComfyUI/nodes.py
MAX_RESOLUTION=16384


class EZCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZCheckpoint",
            display_name="EZ Checkpoint",
            category="prompt_helpers",
            description="Loads a checkpoint, sets clip skip, and applies Loras from the JSON.",
            inputs=[
                io.Combo.Input("checkpoint", options=folder_paths.get_filename_list("checkpoints"), tooltip="The name of the checkpoint (model) to load."),
                io.Int.Input("clip_skip", default=2, min=0, max=24, step=1),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
                io.Vae.Output(),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, checkpoint, clip_skip) -> io.NodeOutput:
        graph = GraphBuilder()

        load_checkpoint = graph.node(
            "CheckpointLoaderSimple",
            ckpt_name=checkpoint,
        )

        clip = load_checkpoint.out(1)

        if clip_skip > 0:
            clip = graph.node(
                "CLIPSetLastLayer",
                clip=clip,
                stop_at_clip_layer=-clip_skip,
            ).out(0)

        return io.NodeOutput(load_checkpoint.out(0), clip, load_checkpoint.out(2), expand=graph.finalize())


class EZBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZBatch",
            display_name="EZ Batch",
            category="prompt_helpers/image",
            description="Generates multiple images in a single run.",
            inputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Input("image_settings", display_name="IMAGE"),
                io.Int.Input("batch_size", default=1, min=1, max=64, tooltip="The number of latent images in the batch."),
                io.Int.Input("select_index", default=-1, min=-1, max=63, tooltip="Generate only one image in the batch. If this is -1 it will generate the entire batch."),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image_settings, batch_size, select_index) -> io.NodeOutput:
        image_settings = image_settings.copy()
        image_settings["batch_size"] = batch_size
        image_settings["select_index"] = select_index
        return io.NodeOutput(image_settings)


class EZBlank(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZBlank",
            display_name="EZ txt2img",
            category="prompt_helpers/image",
            description="Generates a random image.",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=MAX_RESOLUTION, step=8, tooltip="The width of the images in pixels."),
                io.Int.Input("height", default=1024, min=16, max=MAX_RESOLUTION, step=8, tooltip="The height of the images in pixels."),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, width, height) -> io.NodeOutput:
        return io.NodeOutput({
            "type": "BLANK",
            "width": width,
            "height": height,
            "crop": None,
            "detail": None,
            "batch_size": 1,
            "select_index": -1,
        })


class EZImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZImage",
            display_name="EZ img2img",
            category="prompt_helpers/image",
            description="Generates a new image based on an existing image.",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True, tooltip="Only generates inside of the masked region."),
                io.Float.Input("image_weight", default=0.0, min=0.0, max=1.0, step=0.1, round=0.01, tooltip="How much the image should influence the result. Higher number means it closely matches the image, lower number means more random."),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image, image_weight, mask=None) -> io.NodeOutput:
        return io.NodeOutput({
            "type": "IMAGE",
            "image": image,
            "mask": mask,
            "image_weight": image_weight,
            "crop": None,
            "detail": None,
            "batch_size": 1,
            "select_index": -1,
        })


class EZDetail(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZDetail",
            display_name="EZ Detail",
            category="prompt_helpers/image",
            description="Generates more details, very useful for faces and eyes.",
            inputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Input("image_settings", display_name="IMAGE"),

                io.Float.Input("multiplier", default=2.00, min=0.01, max=8.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles size, 0.5 halves size)."),

                io.Combo.Input(
                    "scale_method",
                    options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    default="lanczos",
                    tooltip="Interpolation algorithm. 'area' is best for downscaling, 'lanczos' for upscaling, 'nearest-exact' for pixel art.",
                ),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image_settings, multiplier, scale_method) -> io.NodeOutput:
        image_settings = image_settings.copy()
        image_settings["detail"] = Detail(multiplier, scale_method)
        return io.NodeOutput(image_settings)


class EZCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZCrop",
            display_name="EZ Crop",
            category="prompt_helpers/image",
            description="Crops the image + mask before generating.",
            inputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Input("image_settings", display_name="IMAGE"),
                io.BoundingBox.Input("crop_region"),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image_settings, crop_region) -> io.NodeOutput:
        width = crop_region.get("width", 0)
        height = crop_region.get("height", 0)

        if width > 0 and height > 0:
            x = crop_region.get("x", 0)
            y = crop_region.get("y", 0)

            image_settings = image_settings.copy()
            image_settings["crop"] = Crop(x, x + width, y, y + height)

        return io.NodeOutput(image_settings)


class EZUpscaleTiled(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZUpscaleTiled",
            display_name="EZ Upscale (Tiled)",
            category="prompt_helpers/image",
            description="Generates a new image based on an existing image, but upscaled to a higher resolution.",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("image_weight", default=0.6, min=0.0, max=1.0, step=0.1, round=0.01, tooltip="How much the image should influence the result. Higher number means it closely matches the image, lower number means more random."),
                io.Float.Input("multiplier", default=2.00, min=0.01, max=8.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles size, 0.5 halves size)."),
                io.Int.Input("tile_size", default=1024, min=16, max=MAX_RESOLUTION, step=8, tooltip="The pixel width and height of each tile."),
                io.Int.Input("tile_overlap", default=64, min=0, max=MAX_RESOLUTION, step=1, tooltip="Amount of pixels that overlap between each tile."),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image, image_weight, multiplier, tile_size, tile_overlap) -> io.NodeOutput:
        return io.NodeOutput({
            "type": "UPSCALE_TILED",
            "image": image,
            "image_weight": image_weight,
            "multiplier": multiplier,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "crop": None,
            "detail": None,
            "batch_size": 1,
            "select_index": -1,
        })


class EZPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZPrompt",
            display_name="EZ Prompt",
            category="prompt_helpers",
            description="Guides the image generation with a JSON prompt.",
            inputs=[
                JSON.Input("json", tooltip="The conditioning describing the attributes of the image."),
                io.Float.Input("weight", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01, tooltip="How strongly the prompt should affect the image."),
            ],
            outputs=[
                io.Custom("EZ_PROMPT_SETTINGS").Output(display_name="PROMPT"),
            ],
        )

    @classmethod
    def execute(cls, json, weight) -> io.NodeOutput:
        return io.NodeOutput({
            "json": json,
            "weight": weight,
        })


class EZControlNet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZControlNet",
            display_name="EZ ControlNet",
            category="prompt_helpers/controlnet",
            description="Guides the image generation with a ControlNet.",
            inputs=[
                io.ControlNet.Input("control_net"),
                io.Image.Input("image"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Custom("EZ_CONTROL_NET").Output(display_name="CONTROL_NET"),
            ],
        )

    @classmethod
    def execute(cls, control_net, image, strength, start_percent, end_percent) -> io.NodeOutput:
        return io.NodeOutput([{
            "control_net": control_net,
            "image": image,
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }])


class EmptyControlNet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EmptyControlNet",
            display_name="Empty ControlNet",
            category="prompt_helpers/controlnet",
            description="ControlNet which does nothing.",
            inputs=[],
            outputs=[
                io.Custom("EZ_CONTROL_NET").Output(display_name="CONTROL_NET"),
            ],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput([])


class ConcatenateControlNet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplatePrefix(input=io.Custom("EZ_CONTROL_NET").Input("control_net"), prefix="control_net", min=1, max=20)

        return io.Schema(
            node_id="prompt_helpers: ConcatenateControlNet",
            display_name="Concatenate ControlNet",
            category="prompt_helpers/controlnet",
            description="Combines multiple ControlNets into one ControlNet.",
            inputs=[
                io.Autogrow.Input("control_nets", template=template),
            ],
            outputs=[
                io.Custom("EZ_CONTROL_NET").Output(display_name="CONTROL_NET"),
            ],
        )

    @classmethod
    def execute(cls, control_nets) -> io.NodeOutput:
        inputs = control_nets.values()
        flattened = [x for inner in inputs for x in inner]
        return io.NodeOutput(flattened)


class EZSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZSampler",
            display_name="EZ Sampler",
            category="prompt_helpers",
            description="Settings for how the image generation should happen.",
            inputs=[
                io.Combo.Input("sampler_name", options=sorted(comfy.samplers.KSampler.SAMPLERS), default="euler", tooltip="The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."),
                io.Combo.Input("scheduler", options=sorted(["align_your_steps", "gits", "sdturbo"] + comfy.samplers.KSampler.SCHEDULERS), default="normal", tooltip="The scheduler controls how noise is gradually removed to form the image."),
                io.Int.Input("steps", default=30, min=1, max=10000, tooltip="The number of steps used in the denoising process."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="The random seed used for creating the noise."),
            ],
            outputs=[
                io.Custom("EZ_SAMPLER_SETTINGS").Output(display_name="SAMPLER"),
            ],
        )

    @classmethod
    def execute(cls, sampler_name, scheduler, steps, seed) -> io.NodeOutput:
        return io.NodeOutput({
            "seed": seed,
            "steps": steps,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        })


class EZFilename(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZFilename",
            display_name="EZ Filename",
            description="Generates a filename.",
            category="prompt_helpers",
            inputs=[
                io.AnyType.Input("trigger", tooltip="When this input changes, it will re-run the EZ Filename."),
                io.String.Input("folder", default="", tooltip="The folder that the images will be saved in."),
                io.String.Input("filename", default="%timestamp%", tooltip="The filename for the images.\n\n  %timestamp% is a UTC timestamp when the image was generated"),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, trigger, folder, filename) -> io.NodeOutput:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")

        # TODO make this work on Windows too
        prefix = folder + "/" + filename

        prefix = prefix.replace("%timestamp%", timestamp)

        return io.NodeOutput(prefix)


class EZGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZGenerate",
            display_name="EZ Generate",
            category="prompt_helpers",
            description="Generates the image.",
            inputs=[
                io.Model.Input("model", tooltip="The model used for denoising the input latent."),
                io.Clip.Input("clip", tooltip="The CLIP model used for encoding the text."),
                io.Vae.Input("vae"),

                io.Custom("EZ_PROMPT_SETTINGS").Input("prompt"),
                io.Custom("EZ_SAMPLER_SETTINGS").Input("sampler"),
                io.Custom("EZ_IMAGE_SETTINGS").Input("image"),
                io.Custom("EZ_CONTROL_NET").Input("control_net", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="FULL", tooltip="The full output image."),
                io.Image.Output(display_name="PARTIAL", tooltip="The output image, but cropped and masked."),
            ],
            enable_expand=True,
        )


    @staticmethod
    def apply_controlnet(graph, vae, process, control_net, positive, negative):
        if control_net is not None:
            for item in control_net:
                image = process.apply_to_image(graph, item["image"])

                apply_controlnet = graph.node(
                    "ControlNetApplyAdvanced",
                    positive=positive,
                    negative=negative,
                    control_net=item["control_net"],
                    image=image,
                    strength=item["strength"],
                    start_percent=item["start_percent"],
                    end_percent=item["end_percent"],
                    vae=vae,
                )

                positive = apply_controlnet.out(0)
                negative = apply_controlnet.out(1)

        return (positive, negative)


    @staticmethod
    def encode_prompts(graph, clip, positive, negative):
        # Combines the chunks together with ConditioningConcat
        positive = fold(positive, lambda x, y: graph.node("ConditioningConcat", conditioning_to=x, conditioning_from=y).out(0))
        negative = fold(negative, lambda x, y: graph.node("ConditioningConcat", conditioning_to=x, conditioning_from=y).out(0))

        if positive is None:
            positive = graph.node("CLIPTextEncode", text="", clip=clip).out(0)

        if negative is None:
            negative = graph.node("CLIPTextEncode", text="", clip=clip).out(0)

        return (positive, negative)


    @classmethod
    def convert_prompt(cls, graph, clip, vae, json, process, control_net):
        global_positive = []
        global_negative = []

        region_chunks = []

        for (positive, negative, region) in ProcessJson.iter_chunks(json):
            crop = None
            strength = None

            if region is not None:
                crop = process.region_to_crop(region["x"], region["y"], region["width"], region["height"])

                # Skip regions which are outside of the crop region
                if crop.width() > 0 and crop.height() > 0:
                    strength = region.get("strength", 1.0)
                else:
                    continue

            positive = ProcessJson.serialize_prompts(positive)
            negative = ProcessJson.serialize_prompts(negative)

            if positive is not None:
                positive = graph.node("CLIPTextEncode", text=positive, clip=clip).out(0)

            if negative is not None:
                negative = graph.node("CLIPTextEncode", text=negative, clip=clip).out(0)

            if crop is None:
                if positive is not None:
                    global_positive.append(positive)

                if negative is not None:
                    global_negative.append(negative)

            else:
                chunk = {
                    "positive": [],
                    "negative": [],
                    "crop": crop,
                    "strength": strength,
                }

                if positive is not None:
                    chunk["positive"].append(positive)

                if negative is not None:
                    chunk["negative"].append(negative)

                region_chunks.append(chunk)


        (positive, negative) = cls.encode_prompts(graph, clip, global_positive, global_negative)

        final_positive = [positive]
        final_negative = [negative]


        cropped_mask = process.cropped_mask(graph)

        for chunk in region_chunks:
            # Concats the global prompt with the region prompt
            # If we don't do this then the global prompt will have a weak effect inside the region and it causes artifacting
            chunk["positive"].extend(global_positive)
            chunk["negative"].extend(global_negative)

            (positive, negative) = cls.encode_prompts(graph, clip, chunk["positive"], chunk["negative"])

            positive = process.apply_set_area(graph, cropped_mask, chunk["crop"], chunk["strength"], positive)
            negative = process.apply_set_area(graph, cropped_mask, chunk["crop"], chunk["strength"], negative)

            final_positive.append(positive)
            final_negative.append(negative)


        assert len(final_positive) > 0
        assert len(final_negative) > 0

        # We have to use ConditioningCombine because ConditioningConcat does not work with ConditioningSetMask
        final_positive = fold(final_positive, lambda x, y: graph.node("ConditioningCombine", conditioning_1=x, conditioning_2=y).out(0))
        final_negative = fold(final_negative, lambda x, y: graph.node("ConditioningCombine", conditioning_1=x, conditioning_2=y).out(0))

        (final_positive, final_negative) = cls.apply_controlnet(graph, vae, process, control_net, final_positive, final_negative)

        return (final_positive, final_negative)


    @staticmethod
    def apply_loras(graph, model, clip, json):
        apply_loras = graph.node(
            "prompt_helpers: ApplyLoras",
            model=model,
            clip=clip,
            json=json,
        )

        return (apply_loras.out(0), apply_loras.out(1))


    @classmethod
    def process_json(cls, graph, model, clip, vae, json, process, control_net=None):
        process_json = ProcessJson.process_json(json)

        (model, clip) = cls.apply_loras(graph, model, clip, process_json)

        (positive, negative) = cls.convert_prompt(graph, clip, vae, process_json, process, control_net)

        return (model, clip, positive, negative)


    # https://github.com/Comfy-Org/ComfyUI/blob/b254cecd032e872766965415d120973811e9e360/comfy_extras/nodes_images.py#L573-L574
    @staticmethod
    def get_image_size(image):
        return (image.shape[2], image.shape[1])


    # Everything that isn't masked will be transparent.
    # @TODO Improve this after https://github.com/Comfy-Org/ComfyUI/issues/12580 is fixed
    @staticmethod
    def combine_image_with_mask(graph, image, mask):
        if mask is not None:
            size = graph.node("GetImageSize", image=image)

            mask = graph.node("InvertMask", mask=mask)
            mask = graph.node("MaskToImage", mask=mask.out(0))
            mask = graph.node("RepeatImageBatch", image=mask.out(0), amount=size.out(2))
            mask = graph.node("ImageToMask", image=mask.out(0), channel="red")

            return graph.node(
                "JoinImageWithAlpha",
                image=image,
                alpha=mask.out(0),
            ).out(0)

        else:
            return image


    @staticmethod
    def sampler(graph, model, clip, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        noise = graph.node("RandomNoise", noise_seed=seed)

        #empty = graph.node("CLIPTextEncode", text="", clip=clip)

        guider = graph.node(
            "CFGGuider",
            model=model,
            positive=positive,
            negative=negative,
            cfg=cfg,
        )

        #guider = graph.node(
        #    "PerpNegGuider",
        #    model=model,
        #    positive=positive,
        #    negative=negative,
        #    empty_conditioning=empty.out(0),
        #    cfg=cfg,
        #    neg_scale=neg_scale,
        #)

        sampler = graph.node(
            "KSamplerSelect",
            sampler_name=sampler_name,
        )

        if scheduler == "align_your_steps":
            sigmas = graph.node(
                "AlignYourStepsScheduler",
                model_type="SDXL",
                steps=steps,
                denoise=denoise,
            )

        elif scheduler == "gits":
            sigmas = graph.node(
                "GITSScheduler",
                coeff=1.20,
                steps=steps,
                denoise=denoise,
            )

        elif scheduler == "sdturbo":
            sigmas = graph.node(
                "SDTurboScheduler",
                model=model,
                steps=steps,
                denoise=denoise,
            )

        else:
            sigmas = graph.node(
                "BasicScheduler",
                model=model,
                scheduler=scheduler,
                steps=steps,
                denoise=denoise,
            )

        return graph.node(
            "SamplerCustomAdvanced",
            noise=noise.out(0),
            guider=guider.out(0),
            sampler=sampler.out(0),
            sigmas=sigmas.out(0),
            latent_image=latent_image,
        ).out(1)


    @classmethod
    def generate_text(cls, image, prompt, sampler, control_net=None, **kwargs):
        graph = GraphBuilder()

        process = ProcessImage(
            image["crop"],
            image["detail"],
            image["width"],
            image["height"],
            image["batch_size"],
            image["select_index"],
        )

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], process, control_net)

        empty_image = process.empty_latent(graph)

        sampler = cls.sampler(
            graph=graph,
            model=model,
            clip=clip,
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=prompt["weight"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=positive,
            negative=negative,
            latent_image=empty_image,
            denoise=1.0,
        )

        vae_decode = graph.node(
            "VAEDecode",
            samples=sampler,
            vae=kwargs["vae"],
        )

        resized = process.downscale_image(graph, vae_decode.out(0))

        return io.NodeOutput(
            resized,
            resized,
            expand=graph.finalize(),
        )


    @classmethod
    def generate_image(cls, image, prompt, sampler, control_net=None, **kwargs):
        graph = GraphBuilder()

        original_image = image["image"]
        original_mask = image["mask"]

        (image_width, image_height) = cls.get_image_size(original_image)

        process = ProcessImage(
            image["crop"],
            image["detail"],
            image_width,
            image_height,
            image["batch_size"],
            image["select_index"],
        )

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], process, control_net)

        cropped_mask = process.crop_mask(graph, original_mask)

        if image["image_weight"] == 0.0:
            repeat_latent_batch = process.empty_latent(graph)

        else:
            resized_image = process.apply_to_image(graph, original_image)
            resized_mask = process.resize_mask(graph, cropped_mask)

            if resized_mask is not None:
                # VAEEncodeForInpaint doesn't support image_weight, so we use InpaintModelConditioning instead
                inpaint_model_conditioning = graph.node(
                    "InpaintModelConditioning",
                    positive=positive,
                    negative=negative,
                    vae=kwargs["vae"],
                    pixels=resized_image,
                    mask=resized_mask,
                    noise_mask=True,
                )

                positive = inpaint_model_conditioning.out(0)
                negative = inpaint_model_conditioning.out(1)
                latent_image = inpaint_model_conditioning.out(2)

            else:
                latent_image = graph.node("VAEEncode", pixels=resized_image, vae=kwargs["vae"]).out(0)

            repeat_latent_batch = process.repeat_latent(graph, latent_image)

        sampler = cls.sampler(
            graph=graph,
            model=model,
            clip=clip,
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=prompt["weight"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=positive,
            negative=negative,
            latent_image=repeat_latent_batch,
            denoise=(1.0 - image["image_weight"]),
        )

        vae_decode = graph.node(
            "VAEDecode",
            samples=sampler,
            vae=kwargs["vae"],
        )

        downscaled_image = process.downscale_image(graph, vae_decode.out(0))

        # ComfyUI changes the image even outside of the mask, so we overwrite the image
        # to guarantee that *only* the masked area will be changed
        full_image = process.composite_image(graph, original_image, downscaled_image, cropped_mask)

        return io.NodeOutput(
            full_image,
            cls.combine_image_with_mask(graph, downscaled_image, cropped_mask),
            expand=graph.finalize(),
        )


    @classmethod
    def generate_upscale_tiled(cls, image, prompt, sampler, control_net=None, **kwargs):
        if image["batch_size"] != 1:
            raise RuntimeError("EZ Upscale must have a batch_size of 1")

        if image["detail"] is not None:
            raise RuntimeError("EZ Upscale cannot be combined with EZ Detail")

        graph = GraphBuilder()

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], control_net)

        #model = kwargs["model"]
        #clip = kwargs["clip"]
        #positive = graph.node("CLIPTextEncode", text="", clip=clip).out(0)
        #negative = graph.node("CLIPTextEncode", text="", clip=clip).out(0)

        resized = cls.resize_image(graph, image["image"], image["multiplier"])

        (image_width, image_height) = cls.get_image_size(image["image"])

        # https://github.com/Comfy-Org/ComfyUI/blob/602b2505a4ffeff4a732b8727ce27d3c2a1ef752/comfy_extras/nodes_post_processing.py#L284-L285
        image_width = int(round(image_width * image["multiplier"]))
        image_height = int(round(image_height * image["multiplier"]))

        tiles = []

        # Chunks the image into tiles
        for tile in set(get_image_tiles(image_width, image_height, image["tile_size"], image["tile_overlap"])):
            cropped = graph.node(
                "ImageCrop",
                image=resized,
                x=tile.crop.left,
                y=tile.crop.top,
                width=tile.crop.width(),
                height=tile.crop.height(),
            )

            vae_encode = graph.node("VAEEncode", pixels=cropped.out(0), vae=kwargs["vae"])

            run_sampler = cls.sampler(
                graph=graph,
                model=model,
                clip=clip,
                seed=sampler["seed"],
                steps=sampler["steps"],
                cfg=prompt["weight"],
                sampler_name=sampler["sampler_name"],
                scheduler=sampler["scheduler"],
                positive=positive,
                negative=negative,
                latent_image=vae_encode.out(0),
                denoise=(1.0 - image["image_weight"]),
            )

            vae_decode = graph.node(
                "VAEDecode",
                samples=run_sampler,
                vae=kwargs["vae"],
            )

            tile_mask = graph.node("SolidMask", value=0.0, width=tile.crop.width(), height=tile.crop.height())

            mask = graph.node("SolidMask", value=1.0, width=tile.mask.width(), height=tile.mask.height())

            mask = graph.node("FeatherMask", mask=mask.out(0), left=tile.grow.left, top=tile.grow.top, right=tile.grow.right, bottom=tile.grow.bottom)

            mask = graph.node(
                "MaskComposite",
                destination=tile_mask.out(0),
                source=mask.out(0),
                operation="add",
                x=(tile.mask.left - tile.crop.left),
                y=(tile.mask.top - tile.crop.top),
            )

            tiles.append((tile, vae_decode.out(0), mask.out(0)))

        composited = graph.node("EmptyImage", batch_size=1, color=0, width=image_width, height=image_height).out(0)

        for (tile, image, mask) in tiles:
            #joined = graph.node(
            #    "JoinImageWithAlpha",
            #    image=image,
            #    alpha=graph.node("InvertMask", mask=mask).out(0),
            #).out(0)

            #save_image = graph.node("SaveImage", images=joined, filename_prefix="tmp/tiles/{} {}".format(tile.crop.left, tile.crop.top))

            composited = graph.node(
                "ImageCompositeMasked",
                resize_source=False,
                destination=composited,
                source=image,
                mask=mask,
                x=tile.crop.left,
                y=tile.crop.top,
            ).out(0)


        return io.NodeOutput(
            composited,
            composited,
            expand=graph.finalize(),
        )


    @classmethod
    def execute(cls, image, **kwargs) -> io.NodeOutput:
        if image["type"] == "BLANK":
            return cls.generate_text(image=image, **kwargs)

        elif image["type"] == "IMAGE":
            return cls.generate_image(image=image, **kwargs)

        elif image["type"] == "UPSCALE_TILED":
            return cls.generate_upscale_tiled(image=image, **kwargs)


class EZGenerateSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZGenerateSave",
            display_name="EZ Generate Save",
            category="prompt_helpers",
            description="Generates and saves the image.",
            inputs=[
                io.Model.Input("model", tooltip="The model used for denoising the input latent."),
                io.Clip.Input("clip", tooltip="The CLIP model used for encoding the text."),
                io.Vae.Input("vae"),

                io.String.Input("folder", default="", tooltip="The folder that the images will be saved in."),
                io.String.Input("filename", default="%timestamp%", tooltip="The filename for the images.\n\n  %timestamp% is a UTC timestamp when the image was generated"),

                io.Custom("EZ_PROMPT_SETTINGS").Input("prompt"),
                io.Custom("EZ_SAMPLER_SETTINGS").Input("sampler"),
                io.Custom("EZ_IMAGE_SETTINGS").Input("image"),
                io.Custom("EZ_CONTROL_NET").Input("control_net", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="FULL", tooltip="The full output image."),
                io.Image.Output(display_name="PARTIAL", tooltip="The output image, but cropped and masked."),
                io.String.Output(display_name="PATH", tooltip="The path used for the saved images."),
            ],
            is_output_node=True,
            enable_expand=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        graph = GraphBuilder()

        ez_generate = graph.node("prompt_helpers: EZGenerate", **kwargs)

        filename = graph.node("prompt_helpers: EZFilename", trigger=ez_generate.out(0), folder=kwargs["folder"], filename=kwargs["filename"])

        save_image = graph.node("SaveImage", images=ez_generate.out(0), filename_prefix=filename.out(0))

        return io.NodeOutput(
            ez_generate.out(0),
            ez_generate.out(1),
            filename.out(0),
            expand=graph.finalize(),
        )


class EZNotify(io.ComfyNode):
    notifier = desktop_notifier.DesktopNotifier(
        app_name="ComfyUI",
        app_icon=None,
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZNotify",
            display_name="EZ Notify",
            category="prompt_helpers",
            description="Shows a desktop notification to let you know generation is done.",
            inputs=[
                io.AnyType.Input("trigger", tooltip="When this input changes, it will show the notification."),
                io.String.Input("message", default="Job done", tooltip="The message which will be displayed in the notification."),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, trigger, message) -> io.NodeOutput:
        await cls.notifier.send(
            title="EZ Notify",
            message=message,
            urgency=desktop_notifier.Urgency.Low,
        )

        return io.NodeOutput()
