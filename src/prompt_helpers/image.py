import math
from .utils import clamp


class Crop:
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.top == other.top and self.bottom == other.bottom

    def __hash__(self):
        return hash((self.left, self.right, self.top, self.bottom))


    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top


    def pad(self, horizontal, vertical):
        return Crop(self.left - horizontal, self.right + horizontal, self.top - vertical, self.bottom + vertical)


    def scale(self, multiplier):
        # https://github.com/Comfy-Org/ComfyUI/blob/602b2505a4ffeff4a732b8727ce27d3c2a1ef752/comfy_extras/nodes_post_processing.py#L284-L285
        return Crop(
            int(round(self.left * multiplier)),
            int(round(self.right * multiplier)),
            int(round(self.top * multiplier)),
            int(round(self.bottom * multiplier)),
        )


    def clamp(self, image_width, image_height):
        left = clamp(self.left, 0, image_width)
        right = clamp(self.right, 0, image_width)

        top = clamp(self.top, 0, image_height)
        bottom = clamp(self.bottom, 0, image_height)

        if right < left:
            right = left

        if bottom < top:
            bottom = top

        return Crop(left, right, top, bottom)


    def clamp_to_parent(self, parent):
        left = clamp(self.left, parent.left, parent.right)
        right = clamp(self.right, parent.left, parent.right)

        top = clamp(self.top, parent.top, parent.bottom)
        bottom = clamp(self.bottom, parent.top, parent.bottom)

        if right < left:
            right = left

        if bottom < top:
            bottom = top

        return Crop(left, right, top, bottom)


    def apply_to_image(self, graph, image):
        return graph.node("ImageCrop", image=image, x=self.left, y=self.top, width=self.width(), height=self.height()).out(0)

    def apply_to_mask(self, graph, mask):
        return graph.node("CropMask", mask=mask, x=self.left, y=self.top, width=self.width(), height=self.height()).out(0)


class Detail:
    def __init__(self, resize_type, scale_method):
        self.resize_type = resize_type
        self.scale_method = scale_method


    def is_noop(self):
        return self.resize_type["resize_type"] == "scale by multiplier" and self.resize_type["multiplier"] == 1.0


    def scale(self, width, height):
        # https://github.com/Comfy-Org/ComfyUI/blob/7d437687c260df7772c603658111148e0e863e59/comfy_extras/nodes_post_processing.py#L281-L289
        if self.resize_type["resize_type"] == "scale by multiplier":
            width = int(round(width * self.resize_type["multiplier"]))
            height = int(round(height * self.resize_type["multiplier"]))

        # https://github.com/Comfy-Org/ComfyUI/blob/7d437687c260df7772c603658111148e0e863e59/comfy_extras/nodes_post_processing.py#L346-L357
        elif self.resize_type["resize_type"] == "scale total pixels":
            total = self.resize_type["megapixels"] * 1024 * 1024

            scale_by = math.sqrt(total / (width * height))

            width = int(round(width * scale_by))
            height = int(round(height * scale_by))

        # https://github.com/Comfy-Org/ComfyUI/blob/7d437687c260df7772c603658111148e0e863e59/comfy_extras/nodes_post_processing.py#L306-L324
        elif self.resize_type["resize_type"] == "scale longer dimension":
            largest_size = self.resize_type["longer_size"]

            if height > width:
                width = int(round((width / height) * largest_size))
                height = largest_size
            elif width > height:
                height = int(round((height / width) * largest_size))
                width = largest_size
            else:
                height = largest_size
                width = largest_size

        return (width, height)


    def apply_to_image(self, graph, image):
        if self.is_noop():
            return image

        else:
            # TODO replace with ResizeImageMaskNode after https://github.com/Comfy-Org/ComfyUI/issues/12566 is fixed
            if self.resize_type["resize_type"] == "scale by multiplier":
                return graph.node(
                    "ImageScaleBy",
                    image=image,
                    scale_by=self.resize_type["multiplier"],
                    upscale_method=self.scale_method,
                ).out(0)

            elif self.resize_type["resize_type"] == "scale total pixels":
                return graph.node(
                    "ImageScaleToTotalPixels",
                    image=image,
                    megapixels=self.resize_type["megapixels"],
                    upscale_method=self.scale_method,
                    resolution_steps=1,
                ).out(0)

            elif self.resize_type["resize_type"] == "scale longer dimension":
                return graph.node(
                    "ImageScaleToMaxDimension",
                    image=image,
                    largest_size=self.resize_type["longer_size"],
                    upscale_method=self.scale_method,
                ).out(0)


    def apply_to_mask(self, graph, mask):
        if self.is_noop():
            return mask

        else:
            image = graph.node("MaskToImage", mask=mask).out(0)

            resized = self.apply_to_image(graph, image)

            return graph.node("ImageToMask", image=resized, channel="red").out(0)


class ProcessImage:
    def __init__(self, crop, detail, width, height, batch_size, select_index):
        self.cached_crop_mask = None

        if crop is not None:
            self.crop = crop.clamp(width, height)
        else:
            self.crop = None

        self.detail = detail
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.select_index = select_index


    def with_crop(self, crop):
        return ProcessImage(crop, self.detail, self.width, self.height, self.batch_size, self.select_index)


    def image_crop(self):
        if self.crop:
            return self.crop
        else:
            return Crop(0, self.width, 0, self.height)


    def empty_latent(self, graph):
        crop = self.image_crop()

        width = crop.width()
        height = crop.height()

        if self.detail is not None:
            (width, height) = self.detail.scale(width, height)

        latent_image = graph.node("EmptyLatentImage", width=width, height=height, batch_size=self.batch_size).out(0)

        if self.select_index > -1:
            latent_image = graph.node("LatentFromBatch", samples=latent_image, batch_index=self.select_index, length=1).out(0)

        return latent_image


    def apply_to_image(self, graph, image):
        if self.crop is not None:
            image = self.crop.apply_to_image(graph, image)

        if self.detail is not None:
            image = self.detail.apply_to_image(graph, image)

        return image


    def downscale_image(self, graph, image):
        if self.detail is None or self.detail.is_noop():
            return image

        else:
            crop = self.image_crop()

            return graph.node(
                "ImageScale",
                image=image,
                width=crop.width(),
                height=crop.height(),
                crop="center",
                upscale_method="area",
            ).out(0)


    def repeat_latent(self, graph, latent):
        if self.batch_size == 1:
            return latent

        else:
            repeat_latent_batch = graph.node("RepeatLatentBatch", samples=latent, amount=self.batch_size).out(0)

            if self.select_index > -1:
                repeat_latent_batch = graph.node("LatentFromBatch", samples=repeat_latent_batch, batch_index=self.select_index, length=1).out(0)

            return repeat_latent_batch


    def repeat_image(self, graph, image):
        if self.batch_size == 1 or self.select_index > -1:
            return image

        else:
            return graph.node(
                "RepeatImageBatch",
                image=image,
                amount=self.batch_size,
            ).out(0)


    def crop_mask(self, graph, mask):
        if mask is not None and self.crop is not None:
            mask = self.crop.apply_to_mask(graph, mask)

        return mask


    def resize_mask(self, graph, mask):
        if mask is not None and self.detail is not None:
            mask = self.detail.apply_to_mask(graph, mask)

        return mask


    def composite_image(self, graph, original_image, downscaled_image, cropped_mask):
        if cropped_mask is None and self.crop is None:
            return downscaled_image

        else:
            image_crop = self.image_crop()

            return graph.node(
                "ImageCompositeMasked",
                destination=self.repeat_image(graph, original_image),
                source=downscaled_image,
                mask=cropped_mask,
                x=image_crop.left,
                y=image_crop.top,
                resize_source=False,
            ).out(0)


    def region_to_crop(self, region):
        left = region.x.evaluate_int(self.width)
        right = left + region.width.evaluate_int(self.width)

        top = region.y.evaluate_int(self.height)
        bottom = top + region.height.evaluate_int(self.height)

        (x_feather, y_feather) = region.evaluate_feather(self.width, self.height)

        crop = Crop(left, right, top, bottom).pad(x_feather, y_feather)

        clamped = crop.clamp_to_parent(self.image_crop())

        if clamped.width() == 0 or clamped.height() == 0:
            return None
        else:
            return crop


    def cropped_mask(self, graph):
        if self.cached_crop_mask is None:
            image_crop = self.image_crop()

            self.cached_crop_mask = graph.node(
                "SolidMask",
                value=0.0,
                width=image_crop.width(),
                height=image_crop.height(),
            ).out(0)

        return self.cached_crop_mask


    def apply_set_mask(graph, mask, strength, isolated, conditioning):
        if isolated:
            cond_area = "mask bounds"
        else:
            cond_area = "default"

        return graph.node(
            "ConditioningSetMask",
            conditioning=conditioning,
            mask=mask,
            strength=strength,
            set_cond_area=cond_area,
        ).out(0)


    def apply_set_area(self, graph, region, crop, conditioning):
        cropped_mask = self.cropped_mask()
        image_crop = self.image_crop()

        (x_feather, y_feather) = region.evaluate_feather(self.width, self.height)

        # Crop which contains only solid white pixels
        solid_crop = crop.pad(-x_feather, -y_feather).clamp_to_parent(image_crop)

        # Region occupies the entire image, no need for masking
        if solid_crop == image_crop and region.strength == 1.0:
            return conditioning

        else:
            mask = graph.node(
                "SolidMask",
                value=1.0,
                width=crop.width(),
                height=crop.height(),
            ).out(0)

            if x_feather > 0 or y_feather > 0:
                mask = graph.node("FeatherMask", mask=mask, left=x_feather, top=y_feather, right=x_feather, bottom=y_feather).out(0)

            clamped = crop.clamp_to_parent(image_crop)

            assert clamped.width() > 0
            assert clamped.height() > 0

            if clamped != crop:
                x = clamped.left - crop.left
                y = clamped.top - crop.top

                assert x >= 0
                assert y >= 0

                mask = graph.node(
                    "CropMask",
                    mask=mask,
                    x=x,
                    y=y,
                    width=clamped.width(),
                    height=clamped.height(),
                ).out(0)

            if clamped != image_crop:
                mask = graph.node(
                    "MaskComposite",
                    destination=cropped_mask,
                    source=mask,
                    x=clamped.left - image_crop.left,
                    y=clamped.top - image_crop.top,
                    operation="add",
                ).out(0)

            return ProcessImage.apply_set_mask(graph, mask, region.strength, region.isolated, conditioning)
