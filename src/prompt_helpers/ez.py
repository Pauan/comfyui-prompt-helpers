from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import comfy
import datetime


# Copied from Comfy-Org/ComfyUI/nodes.py
MAX_RESOLUTION=16384


class EZBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZBatch",
            display_name="EZ Batch",
            category="prompt_helpers/image",
            description="Generates multiple images in a single run.",
            inputs=[
                io.Custom("IMAGE_SETTINGS").Input("image_settings", display_name="IMAGE"),
                io.Int.Input("batch_size", default=1, min=1, max=64, tooltip="The number of latent images in the batch."),
                io.Int.Input("select_index", default=-1, min=-1, max=63, tooltip="Generate only one image in the batch. If this is -1 it will generate the entire batch."),
            ],
            outputs=[
                io.Custom("IMAGE_SETTINGS").Output(display_name="IMAGE"),
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
                io.Custom("IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, width, height) -> io.NodeOutput:
        return io.NodeOutput({
            "type": "BLANK",
            "width": width,
            "height": height,
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
                io.Custom("IMAGE_SETTINGS").Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image, image_weight) -> io.NodeOutput:
        return io.NodeOutput({
            "type": "IMAGE",
            "image": image,
            "image_weight": image_weight,
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
                io.Custom("IMAGE_SETTINGS").Output(display_name="IMAGE"),
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
            description="Guides the image generation with a text prompt.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="The conditioning describing the attributes you want to include in the image."),
                io.Float.Input("positive_weight", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01, tooltip="How strongly the positive prompt should affect the image."),

                io.Conditioning.Input("negative", tooltip="The conditioning describing the attributes you want to exclude from the image."),
                io.Float.Input("negative_weight", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01, tooltip="How strongly the negative prompt should affect the image."),
            ],
            outputs=[
                io.Custom("PROMPT_SETTINGS").Output(display_name="PROMPT"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, positive_weight, negative_weight) -> io.NodeOutput:
        return io.NodeOutput({
            "positive": positive,
            "negative": negative,
            "positive_weight": positive_weight,
            "negative_weight": negative_weight,
        })


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
                io.Custom("SAMPLER_SETTINGS").Output(display_name="SAMPLER"),
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
            inputs=[
                io.Image.Input("image"),
                io.String.Input("folder"),
                io.String.Input("filename"),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, folder, filename) -> io.NodeOutput:
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
            description="Generates and saves the image.",
            inputs=[
                io.Model.Input("model", tooltip="The model used for denoising the input latent."),
                io.Clip.Input("clip", tooltip="The CLIP model used for encoding the text."),
                io.Vae.Input("vae"),

                io.String.Input("folder", default="", tooltip="The folder that the images will be saved in."),
                io.String.Input("filename", default="%timestamp%", tooltip="The filename for the images.\n\n  %timestamp% is a UTC timestamp when the image was generated"),

                io.Custom("IMAGE_SETTINGS").Input("image"),
                io.Custom("PROMPT_SETTINGS").Input("prompt"),
                io.Custom("SAMPLER_SETTINGS").Input("sampler"),
            ],
            outputs=[
                io.Image.Output(),
                io.String.Output(),
            ],
            is_output_node=True,
            enable_expand=True,
        )


    @staticmethod
    def sampler(graph, model, clip, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, neg_scale):
        noise = graph.node("RandomNoise", noise_seed=seed)

        empty = graph.node("CLIPTextEncode", text="", clip=clip)

        guider = graph.node(
            "PerpNegGuider",
            model=model,
            positive=positive,
            negative=negative,
            empty_conditioning=empty.out(0),
            cfg=cfg,
            neg_scale=neg_scale,
        )

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
        )


    @classmethod
    def generate_text(cls, image, prompt, sampler, **kwargs):
        graph = GraphBuilder()

        empty_image = graph.node("EmptyLatentImage", width=image["width"], height=image["height"], batch_size=image["batch_size"])

        if image["select_index"] > -1:
            empty_image = graph.node("LatentFromBatch", samples=empty_image.out(0), batch_index=image["select_index"], length=1)

        sampler = cls.sampler(
            graph=graph,
            model=kwargs["model"],
            clip=kwargs["clip"],
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=prompt["positive_weight"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=prompt["positive"],
            negative=prompt["negative"],
            latent_image=empty_image.out(0),
            denoise=1.0,
            neg_scale=prompt["negative_weight"],
        )

        vae_decode = graph.node(
            "VAEDecode",
            samples=sampler.out(0),
            vae=kwargs["vae"],
        )

        filename = graph.node("prompt_helpers: EZFilename", image=vae_decode.out(0), folder=kwargs["folder"], filename=kwargs["filename"])

        save_image = graph.node("SaveImage", images=vae_decode.out(0), filename_prefix=filename.out(0))

        return io.NodeOutput(
            vae_decode.out(0),
            filename.out(0),
            expand=graph.finalize(),
        )


    @classmethod
    def generate_image(cls, image, prompt, sampler, **kwargs):
        graph = GraphBuilder()

        vae_encode = graph.node("VAEEncode", pixels=image["image"], vae=kwargs["vae"])

        if image["batch_size"] == 1:
            repeat_latent_batch = vae_encode

        else:
            repeat_latent_batch = graph.node("RepeatLatentBatch", samples=vae_encode.out(0), amount=image["batch_size"])

            if image["select_index"] > -1:
                repeat_latent_batch = graph.node("LatentFromBatch", samples=repeat_latent_batch.out(0), batch_index=image["select_index"], length=1)

        sampler = cls.sampler(
            graph=graph,
            model=kwargs["model"],
            clip=kwargs["clip"],
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=prompt["positive_weight"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=prompt["positive"],
            negative=prompt["negative"],
            latent_image=repeat_latent_batch.out(0),
            denoise=(1.0 - image["image_weight"]),
            neg_scale=prompt["negative_weight"],
        )

        vae_decode = graph.node(
            "VAEDecode",
            samples=sampler.out(0),
            vae=kwargs["vae"],
        )

        filename = graph.node("prompt_helpers: EZFilename", image=vae_decode.out(0), folder=kwargs["folder"], filename=kwargs["filename"])

        save_image = graph.node("SaveImage", images=vae_decode.out(0), filename_prefix=filename.out(0))

        return io.NodeOutput(
            vae_decode.out(0),
            filename.out(0),
            expand=graph.finalize(),
        )


    @classmethod
    def generate_inpainting(cls, image, prompt, sampler, **kwargs):
        graph = GraphBuilder()

        grow_mask = graph.node("GrowMask", mask=image["mask"], expand=image["grow_mask"], tapered_corners=True)

        # VAEEncodeForInpaint doesn't support image_weight, so we use InpaintModelConditioning instead
        inpaint_model_conditioning = graph.node(
            "InpaintModelConditioning",
            positive=prompt["positive"],
            negative=prompt["negative"],
            vae=kwargs["vae"],
            pixels=image["image"],
            mask=grow_mask.out(0),
            noise_mask=True,
        )

        if image["batch_size"] == 1:
            repeat_latent_batch = inpaint_model_conditioning.out(2)

        else:
            repeat_latent_batch = graph.node("RepeatLatentBatch", samples=inpaint_model_conditioning.out(2), amount=image["batch_size"]).out(0)

            if image["select_index"] > -1:
                repeat_latent_batch = graph.node("LatentFromBatch", samples=repeat_latent_batch, batch_index=image["select_index"], length=1).out(0)

        sampler = cls.sampler(
            graph=graph,
            model=kwargs["model"],
            clip=kwargs["clip"],
            seed=sampler["seed"],
            steps=sampler["steps"],
            cfg=prompt["positive_weight"],
            sampler_name=sampler["sampler_name"],
            scheduler=sampler["scheduler"],
            positive=inpaint_model_conditioning.out(0),
            negative=inpaint_model_conditioning.out(1),
            latent_image=repeat_latent_batch,
            denoise=(1.0 - image["image_weight"]),
            neg_scale=prompt["negative_weight"],
        )

        vae_decode = graph.node(
            "VAEDecode",
            samples=sampler.out(0),
            vae=kwargs["vae"],
        )

        if image["batch_size"] == 1 or image["select_index"] > -1:
            repeat_image_batch = image["image"]

        else:
            repeat_image_batch = graph.node(
                "RepeatImageBatch",
                image=image["image"],
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

        filename = graph.node("prompt_helpers: EZFilename", image=image_composite_masked.out(0), folder=kwargs["folder"], filename=kwargs["filename"])

        save_image = graph.node("SaveImage", images=image_composite_masked.out(0), filename_prefix=filename.out(0))

        return io.NodeOutput(
            image_composite_masked.out(0),
            filename.out(0),
            expand=graph.finalize(),
        )


    @classmethod
    def execute(cls, image, prompt, sampler, **kwargs) -> io.NodeOutput:
        if image["type"] == "BLANK":
            return cls.generate_text(image=image, prompt=prompt, sampler=sampler, **kwargs)

        elif image["type"] == "IMAGE":
            return cls.generate_image(image=image, prompt=prompt, sampler=sampler, **kwargs)

        elif image["type"] == "INPAINT":
            return cls.generate_inpainting(image=image, prompt=prompt, sampler=sampler, **kwargs)
