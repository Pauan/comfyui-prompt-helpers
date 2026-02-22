from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import comfy
import folder_paths
import datetime
from .prompt import JSON
from .upscale import get_image_tiles


# Copied from Comfy-Org/ComfyUI/nodes.py
MAX_RESOLUTION=16384

DEFAULT_SCALING = "lanczos"


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
                io.Int.Input("clip_skip", default=-2, min=-24, max=-1, step=1),
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

        set_last_layer = graph.node(
            "CLIPSetLastLayer",
            clip=load_checkpoint.out(1),
            stop_at_clip_layer=clip_skip,
        )

        return io.NodeOutput(load_checkpoint.out(0), set_last_layer.out(0), load_checkpoint.out(2), expand=graph.finalize())


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
            "resize_multiplier": 1,
            "scale_method": DEFAULT_SCALING,
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
                io.Float.Input("image_weight", default=0.0, min=0.0, max=1.0, step=0.1, round=0.01, tooltip="How much the image should influence the result. Higher number means it closely matches the image, lower number means more random."),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image, image_weight) -> io.NodeOutput:
        return io.NodeOutput({
            "type": "IMAGE",
            "image": image,
            "image_weight": image_weight,
            "resize_multiplier": 1,
            "scale_method": DEFAULT_SCALING,
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
                    default=DEFAULT_SCALING,
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
        image_settings["resize_multiplier"] = multiplier
        image_settings["scale_method"] = scale_method
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
            "resize_multiplier": 1,
            "scale_method": DEFAULT_SCALING,
            "batch_size": 1,
            "select_index": -1,
        })


class EZInpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZInpaint",
            display_name="EZ Inpaint",
            category="prompt_helpers/image",
            description="Generates a new image based on an existing image, but only in the masked area.",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("image_weight", default=0.0, min=0.0, max=1.0, step=0.1, round=0.01, tooltip="How much the image should influence the result. Higher number means it closely matches the image, lower number means more random."),
                io.Int.Input("grow_mask", default=0, min=-MAX_RESOLUTION, max=MAX_RESOLUTION, step=1),
            ],
            outputs=[
                io.Custom("EZ_IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image, mask, image_weight, grow_mask) -> io.NodeOutput:
        return io.NodeOutput({
            "type": "INPAINT",
            "image": image,
            "mask": mask,
            "image_weight": image_weight,
            "grow_mask": grow_mask,
            "resize_multiplier": 1,
            "scale_method": DEFAULT_SCALING,
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
                io.Image.Output(display_name="MASKED", tooltip="The output image, but the non-masked areas are transparent."),
                io.Image.Output(display_name="FULL", tooltip="The full output image."),
            ],
            enable_expand=True,
        )


    @staticmethod
    def convert_prompt(graph, clip, vae, control_net, json):
        from_json = graph.node("prompt_helpers: FromJSON", clip=clip, json=json)

        positive = from_json.out(0)
        negative = from_json.out(1)

        if control_net:
            for item in control_net:
                apply_controlnet = graph.node(
                    "ControlNetApplyAdvanced",
                    positive=positive,
                    negative=negative,
                    control_net=item["control_net"],
                    image=item["image"],
                    strength=item["strength"],
                    start_percent=item["start_percent"],
                    end_percent=item["end_percent"],
                    vae=vae,
                )

                positive = apply_controlnet.out(0)
                negative = apply_controlnet.out(1)

        return (positive, negative)


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
    def process_json(cls, graph, model, clip, vae, json, control_net=None):
        process_json = graph.node("prompt_helpers: ProcessJson", json=json).out(0)

        (model, clip) = cls.apply_loras(graph, model, clip, process_json)

        (positive, negative) = cls.convert_prompt(graph, clip, vae, control_net, process_json)

        return (model, clip, positive, negative)


    # https://github.com/Comfy-Org/ComfyUI/blob/b254cecd032e872766965415d120973811e9e360/comfy_extras/nodes_images.py#L573-L574
    @staticmethod
    def get_image_size(image):
        return (image.shape[2], image.shape[1])


    @staticmethod
    def resize_image(graph, image, multiplier, scale_method):
        if multiplier == 1.0:
            return image

        else:
            return graph.node(
                "ImageScaleBy",
                image=image,
                scale_by=multiplier,
                upscale_method=scale_method,
            ).out(0)


    #@staticmethod
    #def resize_mask(graph, mask, multiplier, scale_method):
    #    if multiplier == 1.0:
    #        return mask
    #
    #    else:
    #        resize_type = {
    #            "resize_type": "scale by multiplier",
    #            "multiplier": multiplier,
    #        }
    #
    #        return graph.node(
    #            "ResizeImageMaskNode",
    #            resize_type=resize_type,
    #            input=mask,
    #            scale_method=scale_method,
    #        ).out(0)


    # TODO replace with ResizeImageMaskNode after https://github.com/Comfy-Org/ComfyUI/issues/12566 is fixed
    @classmethod
    def resize_mask(cls, graph, mask, multiplier, scale_method):
        if multiplier == 1.0:
            return mask

        else:
            image = graph.node("MaskToImage", mask=mask).out(0)

            resized = cls.resize_image(graph, image, multiplier, scale_method)

            return graph.node("ImageToMask", image=resized, channel="red").out(0)


    @staticmethod
    def repeat_batch_size(graph, image, batch_size, select_index):
        if image == 1:
            return image

        else:
            repeat_latent_batch = graph.node("RepeatLatentBatch", samples=image, amount=batch_size).out(0)

            if select_index > -1:
                repeat_latent_batch = graph.node("LatentFromBatch", samples=repeat_latent_batch, batch_index=select_index, length=1).out(0)

            return repeat_latent_batch


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

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], control_net)

        empty_image = graph.node(
            "EmptyLatentImage",
            # https://github.com/Comfy-Org/ComfyUI/blob/602b2505a4ffeff4a732b8727ce27d3c2a1ef752/comfy_extras/nodes_post_processing.py#L284-L285
            width=int(round(image["width"] * image["resize_multiplier"])),
            height=int(round(image["height"] * image["resize_multiplier"])),
            batch_size=image["batch_size"],
        ).out(0)

        if image["select_index"] > -1:
            empty_image = graph.node("LatentFromBatch", samples=empty_image, batch_index=image["select_index"], length=1).out(0)

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

        resized = cls.resize_image(graph, vae_decode.out(0), 1.0 / image["resize_multiplier"], image["scale_method"])

        return io.NodeOutput(
            resized,
            resized,
            expand=graph.finalize(),
        )


    @classmethod
    def generate_image(cls, image, prompt, sampler, control_net=None, **kwargs):
        graph = GraphBuilder()

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], control_net)

        resized = cls.resize_image(graph, image["image"], image["resize_multiplier"], image["scale_method"])

        vae_encode = graph.node("VAEEncode", pixels=resized, vae=kwargs["vae"])

        repeat_latent_batch = cls.repeat_batch_size(graph, vae_encode.out(0), image["batch_size"], image["select_index"])

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

        resized = cls.resize_image(graph, vae_decode.out(0), 1.0 / image["resize_multiplier"], image["scale_method"])

        return io.NodeOutput(
            resized,
            resized,
            expand=graph.finalize(),
        )


    @classmethod
    def generate_upscale_tiled(cls, image, prompt, sampler, control_net=None, **kwargs):
        if image["batch_size"] != 1:
            raise RuntimeError("EZ Upscale must have a batch_size of 1")

        graph = GraphBuilder()

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], control_net)

        #model = kwargs["model"]
        #clip = kwargs["clip"]
        #positive = graph.node("CLIPTextEncode", text="", clip=clip).out(0)
        #negative = graph.node("CLIPTextEncode", text="", clip=clip).out(0)

        # TODO support EZ Detail
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
    def generate_inpainting(cls, image, prompt, sampler, control_net=None, **kwargs):
        graph = GraphBuilder()

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], control_net)

        resized = cls.resize_image(graph, image["image"], image["resize_multiplier"], image["scale_method"])

        resized_mask = cls.resize_mask(graph, image["mask"], image["resize_multiplier"], image["scale_method"])

        grow_mask = graph.node("GrowMask", mask=resized_mask, expand=image["grow_mask"], tapered_corners=True)

        # VAEEncodeForInpaint doesn't support image_weight, so we use InpaintModelConditioning instead
        inpaint_model_conditioning = graph.node(
            "InpaintModelConditioning",
            positive=positive,
            negative=negative,
            vae=kwargs["vae"],
            pixels=resized,
            mask=grow_mask.out(0),
            noise_mask=True,
        )

        repeat_latent_batch = cls.repeat_batch_size(graph, inpaint_model_conditioning.out(2), image["batch_size"], image["select_index"])

        sampler = cls.sampler(
            graph=graph,
            model=model,
            clip=clip,
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=prompt["weight"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=inpaint_model_conditioning.out(0),
            negative=inpaint_model_conditioning.out(1),
            latent_image=repeat_latent_batch,
            denoise=(1.0 - image["image_weight"]),
        )

        vae_decode = graph.node(
            "VAEDecode",
            samples=sampler,
            vae=kwargs["vae"],
        )

        if image["batch_size"] == 1 or image["select_index"] > -1:
            repeat_image_batch = resized

        else:
            repeat_image_batch = graph.node(
                "RepeatImageBatch",
                image=resized,
                amount=image["batch_size"],
            ).out(0)

        # ComfyUI changes the image even outside of the mask, so we overwrite the image
        # to guarantee that *only* the masked area will be changed
        image_composite_masked = graph.node(
            "ImageCompositeMasked",
            destination=repeat_image_batch,
            source=vae_decode.out(0),
            mask=grow_mask.out(0),
            x=0,
            y=0,
            resize_source=False,
        )

        # Everything that isn't masked will be transparent.
        invert_mask = graph.node("InvertMask", mask=grow_mask.out(0))

        join_image = graph.node(
            "JoinImageWithAlpha",
            image=vae_decode.out(0),
            alpha=invert_mask.out(0),
        )

        return io.NodeOutput(
            cls.resize_image(graph, join_image.out(0), 1.0 / image["resize_multiplier"], image["scale_method"]),
            cls.resize_image(graph, image_composite_masked.out(0), 1.0 / image["resize_multiplier"], image["scale_method"]),
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

        elif image["type"] == "INPAINT":
            return cls.generate_inpainting(image=image, **kwargs)


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
                io.Image.Output(display_name="MASKED", tooltip="The output image, but the non-masked areas are transparent."),
                io.Image.Output(display_name="FULL", tooltip="The full output image."),
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
