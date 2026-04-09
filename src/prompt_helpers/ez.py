from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
import comfy
import folder_paths
import datetime
import desktop_notifier
from .prompt import (JSON, ProcessedJSON)
from .upscale import get_image_tiles
from .image import (Crop, Detail, ProcessImage)
from .utils import (fold)


# Copied from Comfy-Org/ComfyUI/nodes.py
MAX_RESOLUTION=16384


class EncodePrompts:
    def __init__(self):
        self.seen_prompts = set()
        self.positive = []
        self.negative = []

    def check_prompt(self, prompt):
        if prompt.prompt in self.seen_prompts:
            raise RuntimeError("Duplicate prompt: {}".format(prompt.prompt))

    def add_prompt(self, prompt):
        self.check_prompt(prompt)
        self.seen_prompts.add(prompt.prompt)


class EncodeRegion(EncodePrompts):
    def __init__(self, region, crop):
        super().__init__()
        self.region = region
        self.crop = crop

    def __hash__(self):
        return hash(("region", self.region.hash_with_crop(self.crop)))

    def isolated_crop(self, process):
        if self.region.isolated:
            # TODO don't rely on internal properties of process
            # TODO is this the right clamping ?
            return self.crop.clamp_to_parent(process.bounds)
        else:
            return None

    def apply(self, graph, process, conditioning):
        return process.apply_set_area(graph, self.region, self.crop, conditioning)


class EncodeMaskRegion(EncodePrompts):
    def __init__(self, region, mask):
        super().__init__()
        self.region = region
        self.mask = mask

    def __hash__(self):
        return hash(("mask-region", self.region))

    def isolated_crop(self, process):
        if self.region.isolated:
            raise RuntimeError("isolated is not implemented yet for MASK-REGION")
        else:
            return None

    def apply(self, graph, process, conditioning):
        return ProcessImage.apply_set_mask(graph, self.mask, self.region.strength, self.region.isolated, conditioning)


class EncodeRegions:
    def __init__(self):
        self.cached_prompts = {}
        self.cached_regions = {}

        self.global_region = EncodePrompts()

        self.regions = []


    def add_region(self, region):
        cached = self.cached_regions.get(region, None)

        if cached is None:
            cached = region
            self.cached_regions[region] = region
            self.regions.append(region)

        return cached


    def add_prompts(self, region, prompts):
        for prompt in prompts:
            if region is not self.global_region:
                self.global_region.check_prompt(prompt)
            region.add_prompt(prompt)


    def encode_prompts(self, graph, clip, prompts):
        text = ProcessedJSON.serialize_prompts(prompts)

        encoded = self.cached_prompts.get(text, None)

        if encoded is None:
            encoded = graph.node("CLIPTextEncode", text=text, clip=clip).out(0)
            self.cached_prompts[text] = encoded

        return encoded


    def encode_chunks(self, graph, clip, region, chunks):
        output = []

        for prompts in chunks:
            assert len(prompts) > 0

            self.add_prompts(region, prompts)
            output.append(self.encode_prompts(graph, clip, prompts))

        return output


    @staticmethod
    def concat_conditions(graph, conditions):
        # Combines the chunks together with ConditioningConcat
        return fold(conditions, lambda x, y: graph.node("ConditioningConcat", conditioning_to=x, conditioning_from=y).out(0))


    @staticmethod
    def combine_conditions(graph, conditions):
        # We have to use ConditioningCombine because ConditioningConcat does not work with ConditioningSetMask
        return fold(conditions, lambda x, y: graph.node("ConditioningCombine", conditioning_1=x, conditioning_2=y).out(0))


    @staticmethod
    def append(list, item):
        if item is not None:
            list.append(item)
        return list


    def apply_controlnet(self, graph, clip, vae, process, control_net, positive, negative):
        if positive is None:
            positive = self.encode_prompts(graph, clip, [])

        if negative is None:
            negative = self.encode_prompts(graph, clip, [])

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


    def get_region(self, mask_regions, process, chunk):
        if chunk.region is not None:
            crop = process.region_to_crop(chunk.region)

            if crop is not None:
                return self.add_region(EncodeRegion(chunk.region, crop))

            # If the region is outside of the crop region then it will be skipped
            else:
                return None

        if chunk.mask_region is not None:
            mask = mask_regions.get(chunk.mask_region.name, None)

            if mask is not None:
                mask = process.apply_to_mask(graph, mask)

                return self.add_region(EncodeMaskRegion(chunk.mask_region, mask))

            else:
                raise RuntimeError("Mask region \"{}\" not found.".format(chunk.mask_region.name))

        return self.global_region


    def process(self, graph, clip, vae, processed_json, mask_regions, process, control_net):
        for chunk in processed_json.chunks:
            region = self.get_region(mask_regions, process, chunk)

            if region is not None:
                region.positive.extend(self.encode_chunks(graph, clip, region, chunk.positive))
                region.negative.extend(self.encode_chunks(graph, clip, region, chunk.negative))


        normal_positive = []
        normal_negative = []

        isolated_positive = []
        isolated_negative = []


        global_positive = EncodeRegions.concat_conditions(graph, self.global_region.positive)
        global_negative = EncodeRegions.concat_conditions(graph, self.global_region.negative)

        if global_positive is not None:
            normal_positive.append(global_positive)

        if global_negative is not None:
            normal_negative.append(global_negative)


        for region in self.regions:
            isolated_crop = region.isolated_crop(process)

            if isolated_crop is not None:
                # Concats the global prompt with the region prompt
                # If we don't do this then the global prompt will have a weak effect inside the region and it causes artifacting
                positive = EncodeRegions.concat_conditions(graph, EncodeRegions.append(region.positive, global_positive))
                negative = EncodeRegions.concat_conditions(graph, EncodeRegions.append(region.negative, global_negative))

                # Crop the controlnet to the isolated region
                (positive, negative) = self.apply_controlnet(graph, clip, vae, process.with_crop(isolated_crop), control_net, positive, negative)

                positive = region.apply(graph, process, positive)
                negative = region.apply(graph, process, negative)

                isolated_positive.append(positive)
                isolated_negative.append(negative)

            else:
                if len(region.positive) > 0:
                    positive = EncodeRegions.concat_conditions(graph, EncodeRegions.append(region.positive, global_positive))

                    assert positive is not None

                    positive = region.apply(graph, process, positive)
                    normal_positive.append(positive)

                if len(region.negative) > 0:
                    negative = EncodeRegions.concat_conditions(graph, EncodeRegions.append(region.negative, global_negative))

                    assert negative is not None

                    negative = region.apply(graph, process, negative)
                    normal_negative.append(negative)


        positive = EncodeRegions.combine_conditions(graph, normal_positive)
        negative = EncodeRegions.combine_conditions(graph, normal_negative)

        (positive, negative) = self.apply_controlnet(graph, clip, vae, process, control_net, positive, negative)

        isolated_positive.append(positive)
        isolated_negative.append(negative)

        return (
            EncodeRegions.combine_conditions(graph, isolated_positive),
            EncodeRegions.combine_conditions(graph, isolated_negative),
        )


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

                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Selects how to resize",
                    options=[
                        io.DynamicCombo.Option("scale total pixels", [
                            io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01, tooltip="Target total megapixels (e.g., 1.0 ≈ 1024×1024). Aspect ratio is preserved."),
                        ]),
                        io.DynamicCombo.Option("scale by multiplier", [
                            io.Float.Input("multiplier", default=2.00, min=0.01, max=8.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles size, 0.5 halves size)."),
                        ]),
                        io.DynamicCombo.Option("scale longer dimension", [
                            io.Int.Input("longer_size", default=1024, min=0, max=MAX_RESOLUTION, step=1, tooltip="The longer edge will be resized to this value. Aspect ratio is preserved."),
                        ]),
                    ],
                ),

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
    def execute(cls, image_settings, resize_type, scale_method) -> io.NodeOutput:
        image_settings = image_settings.copy()
        image_settings["detail"] = Detail(resize_type, scale_method)
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
            category="prompt_helpers/prompt",
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
            "masks": {},
        })


class EZMaskRegions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="prompt_helpers: EZMaskRegions",
            display_name="EZ Mask Regions",
            category="prompt_helpers/prompt",
            description="Sets masks which can be used as a mask region in the JSON prompt.",
            inputs=[
                io.Custom("EZ_PROMPT_SETTINGS").Input("prompt"),
                io.Mask.Input("masks", tooltip="List of masks."),
                io.String.Input("names", tooltip="Names of the masks, in the same order as the masks."),
            ],
            outputs=[
                io.Custom("EZ_PROMPT_SETTINGS").Output(display_name="PROMPT", is_output_list=True),
            ],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, prompt, masks, names) -> io.NodeOutput:
        outputs = []

        for prompt in prompt:
            prompt = prompt.copy()

            mappings = prompt["masks"].copy()

            for (name, mask) in zip(names, masks):
                mappings[name] = mask

            prompt["masks"] = mappings

            outputs.append(prompt)

        return io.NodeOutput(outputs)


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
            category="prompt_helpers/sampler",
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
    def apply_loras(graph, model, clip, processed):
        apply_loras = graph.node(
            "prompt_helpers: ApplyLoras",
            model=model,
            clip=clip,
            loras=processed.loras,
        )

        return (apply_loras.out(0), apply_loras.out(1))


    @classmethod
    def process_json(cls, graph, model, clip, vae, json, mask_regions, process, control_net=None):
        processed_json = ProcessedJSON()
        processed_json.process(json)

        (model, clip) = cls.apply_loras(graph, model, clip, processed_json)

        encoded = EncodeRegions()

        (positive, negative) = encoded.process(graph, clip, vae, processed_json, mask_regions, process, control_net)

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
            size = graph.node("GetImageSize", image=image).out(2)

            mask = graph.node("InvertMask", mask=mask).out(0)
            mask = graph.node("MaskToImage", mask=mask).out(0)
            mask = graph.node("RepeatImageBatch", image=mask, amount=size).out(0)
            mask = graph.node("ImageToMask", image=mask, channel="red").out(0)

            return graph.node(
                "JoinImageWithAlpha",
                image=image,
                alpha=mask,
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

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], prompt["masks"], process, control_net)

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

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], prompt["masks"], process, control_net)

        if image["image_weight"] == 0.0:
            repeat_latent_batch = process.empty_latent(graph)

        else:
            resized_image = process.apply_to_image(graph, original_image)
            resized_mask = process.apply_to_mask(graph, original_mask)

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

        cropped_mask = process.crop_mask(graph, original_mask)

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

        (model, clip, positive, negative) = cls.process_json(graph, kwargs["model"], kwargs["clip"], kwargs["vae"], prompt["json"], prompt["masks"], control_net)

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
