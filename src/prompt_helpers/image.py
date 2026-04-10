import math
from .utils import (clamp, snap_to_increment)

# Stable diffusion currently requires images to be multiplies of 8
INCREMENT = 8

DEFAULT_SCALE_METHOD = "lanczos"
DOWNSCALE_METHOD = "area"

# Image size where it will use VAE tiling
VAE_TILING_THRESHOLD = 1536 * 1536


class Size:
    def __init__(self, width, height):
        assert isinstance(width, int)
        assert isinstance(height, int)

        self.width = width
        self.height = height

    def __eq__(self, other):
        return self.width == other.width and self.height == other.height


    def snap_to_increment(self, increment):
        return Size(snap_to_increment(self.width, increment), snap_to_increment(self.height, increment))


    def scale(self, info):
        width = self.width
        height = self.height

        # https://github.com/Comfy-Org/ComfyUI/blob/7d437687c260df7772c603658111148e0e863e59/comfy_extras/nodes_post_processing.py#L281-L289
        if info["resize_type"] == "scale by multiplier":
            width = int(round(width * info["multiplier"]))
            height = int(round(height * info["multiplier"]))

        # https://github.com/Comfy-Org/ComfyUI/blob/7d437687c260df7772c603658111148e0e863e59/comfy_extras/nodes_post_processing.py#L346-L357
        elif info["resize_type"] == "scale total pixels":
            total = info["megapixels"] * 1024 * 1024

            scale_by = math.sqrt(total / (width * height))

            width = int(round(width * scale_by))
            height = int(round(height * scale_by))

        # https://github.com/Comfy-Org/ComfyUI/blob/7d437687c260df7772c603658111148e0e863e59/comfy_extras/nodes_post_processing.py#L306-L324
        elif info["resize_type"] == "scale longer dimension":
            largest_size = info["longer_size"]

            if height > width:
                width = int(round((width / height) * largest_size))
                height = largest_size
            elif width > height:
                height = int(round((height / width) * largest_size))
                width = largest_size
            else:
                height = largest_size
                width = largest_size

        else:
            raise RuntimeError("Unknown resize_type {}".format(info["resize_type"]))

        return Size(width, height)


    def resize_image(self, graph, image, scale_method):
        # TODO replace with ResizeImageMaskNode after https://github.com/Comfy-Org/ComfyUI/issues/12566 is fixed
        return graph.node(
            "ImageScale",
            image=image,
            width=self.width,
            height=self.height,
            crop="disabled",
            upscale_method=scale_method,
        ).out(0)


    def resize_mask(self, graph, mask, scale_method):
        image = graph.node("MaskToImage", mask=mask).out(0)
        resized = self.resize_image(graph, image, scale_method)
        return graph.node("ImageToMask", image=resized, channel="red").out(0)


class Crop:
    def __init__(self, left, right, top, bottom):
        assert isinstance(left, int)
        assert isinstance(right, int)
        assert isinstance(top, int)
        assert isinstance(bottom, int)

        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.top == other.top and self.bottom == other.bottom

    def __hash__(self):
        return hash((self.left, self.right, self.top, self.bottom))

    def copy(self):
        return Crop(self.left, self.right, self.top, self.bottom)


    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top

    def size(self):
        return Size(self.width(), self.height())


    def pad(self, horizontal, vertical):
        return Crop(self.left - horizontal, self.right + horizontal, self.top - vertical, self.bottom + vertical)


    def relative_to_parent(self, parent):
        clamped = self.clamp_to_parent(parent)

        shift_x = clamped.left - self.left
        shift_y = clamped.top - self.top

        assert shift_x >= 0
        assert shift_y >= 0

        crop = Crop(
            shift_x,
            shift_x + clamped.width(),
            shift_y,
            shift_y + clamped.height(),
        )

        assert crop.left >= 0
        assert crop.right >= 0
        assert crop.top >= 0
        assert crop.bottom >= 0

        assert crop.right >= crop.left
        assert crop.bottom >= crop.top

        assert crop.width() <= parent.width()
        assert crop.height() <= parent.height()

        return crop


    def scale(self, multiplier):
        # https://github.com/Comfy-Org/ComfyUI/blob/602b2505a4ffeff4a732b8727ce27d3c2a1ef752/comfy_extras/nodes_post_processing.py#L284-L285
        return Crop(
            int(round(self.left * multiplier)),
            int(round(self.right * multiplier)),
            int(round(self.top * multiplier)),
            int(round(self.bottom * multiplier)),
        )


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


    def snap_to_increment(self, parent, increment):
        assert self.right >= self.left
        assert self.bottom >= self.top

        width = snap_to_increment(self.right - self.left, increment)
        height = snap_to_increment(self.bottom - self.top, increment)

        # Increases the crop to the right and bottom
        right = clamp(self.left + width, parent.left, parent.right)
        bottom = clamp(self.top + height, parent.top, parent.bottom)

        width = snap_to_increment(right - self.left, increment)
        height = snap_to_increment(bottom - self.top, increment)

        # If the crop was clamped, increases the crop to the left and top
        left = clamp(right - width, parent.left, parent.right)
        top = clamp(bottom - height, parent.top, parent.bottom)

        assert right >= left
        assert bottom >= top

        return Crop(left, right, top, bottom)


    def crop_image(self, graph, image):
        return graph.node(
            "ImageCrop",
            image=image,
            x=self.left,
            y=self.top,
            width=self.width(),
            height=self.height(),
        ).out(0)


    def crop_mask(self, graph, mask):
        return graph.node(
            "CropMask",
            mask=mask,
            x=self.left,
            y=self.top,
            width=self.width(),
            height=self.height(),
        ).out(0)


class Detail:
    def __init__(self, resize_type, scale_method):
        self.resize_type = resize_type
        self.scale_method = scale_method


class ProcessImage:
    def __init__(self, crop, detail, width, height, batch_size, select_index):
        # Bounds for the original image
        self.bounds = Crop(0, width, 0, height)

        # Cropped part of the original image
        if crop is None:
            self.crop = self.bounds.copy()
            self.snapped_crop = self.crop.copy()

        else:
            self.crop = crop
            self.snapped_crop = self.crop.snap_to_increment(self.bounds, INCREMENT)

        # TODO figure out a better solution for allowing crops outside of the bounds
        assert self.crop == self.crop.clamp_to_parent(self.bounds)
        assert self.snapped_crop == self.snapped_crop.clamp_to_parent(self.bounds)

        # Resized size of the cropped part
        if detail is None:
            self.resized_size = self.snapped_crop.size().snap_to_increment(INCREMENT)
            self.scale_method = DEFAULT_SCALE_METHOD
        else:
            self.resized_size = self.snapped_crop.size().scale(detail.resize_type).snap_to_increment(INCREMENT)
            self.scale_method = detail.scale_method

        self.detail = detail
        self.batch_size = batch_size
        self.select_index = select_index
        self.cached_bounds_mask = None


    def with_crop(self, crop):
        return ProcessImage(crop, self.detail, self.bounds.width(), self.bounds.height(), self.batch_size, self.select_index)


    def vae_decode(self, graph, vae, latent):
        if (self.resized_size.width * self.resized_size.height) > VAE_TILING_THRESHOLD:
            return graph.node("VAEDecodeTiled", samples=latent, vae=vae, tile_size=1024, overlap=64, temporal_size=64, temporal_overlap=8).out(0)
        else:
            return graph.node("VAEDecode", samples=latent, vae=vae).out(0)


    def vae_encode(self, graph, vae, image):
        if (self.resized_size.width * self.resized_size.height) > VAE_TILING_THRESHOLD:
            return graph.node("VAEEncodeTiled", pixels=image, vae=vae, tile_size=1024, overlap=64, temporal_size=64, temporal_overlap=8).out(0)
        else:
            return graph.node("VAEEncode", pixels=image, vae=vae).out(0)


    def empty_latent(self, graph):
        assert self.resized_size.width % INCREMENT == 0
        assert self.resized_size.height % INCREMENT == 0

        latent_image = graph.node(
            "EmptyLatentImage",
            width=self.resized_size.width,
            height=self.resized_size.height,
            batch_size=self.batch_size,
        ).out(0)

        if self.select_index > -1:
            latent_image = graph.node("LatentFromBatch", samples=latent_image, batch_index=self.select_index, length=1).out(0)

        return latent_image


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


    def apply_to_image(self, graph, image):
        # TODO resize the image to self.bounds before cropping
        if self.snapped_crop != self.bounds:
            image = self.snapped_crop.crop_image(graph, image)

        if self.resized_size != self.snapped_crop.size():
            image = self.resized_size.resize_image(graph, image, self.scale_method)

        return image


    def apply_to_mask(self, graph, mask):
        if mask is not None:
            # TODO resize the mask to self.bounds before cropping
            if self.snapped_crop != self.bounds:
                mask = self.snapped_crop.crop_mask(graph, mask)

            if self.resized_size != self.snapped_crop.size():
                mask = self.resized_size.resize_mask(graph, mask, self.scale_method)

        return mask


    def downscale_image(self, graph, image):
        if self.resized_size != self.snapped_crop.size():
            image = self.snapped_crop.size().resize_image(graph, image, DOWNSCALE_METHOD)

        if self.snapped_crop != self.crop:
            image = self.snapped_crop.relative_to_parent(self.crop).crop_image(graph, image)

        return image


    def crop_mask(self, graph, mask):
        if mask is not None:
            if self.crop != self.bounds:
                mask = self.crop.crop_mask(graph, mask)

        return mask


    def composite_image(self, graph, original_image, downscaled_image, cropped_mask):
        if cropped_mask is None and self.crop == self.bounds:
            return downscaled_image

        else:
            return graph.node(
                "ImageCompositeMasked",
                destination=self.repeat_image(graph, original_image),
                source=downscaled_image,
                mask=cropped_mask,
                x=self.crop.left,
                y=self.crop.top,
                resize_source=False,
            ).out(0)


    def region_to_crop(self, region):
        width = self.bounds.width()
        height = self.bounds.height()

        left = region.x.evaluate_int(width)
        right = left + region.width.evaluate_int(width)

        top = region.y.evaluate_int(height)
        bottom = top + region.height.evaluate_int(height)

        (x_feather, y_feather) = region.evaluate_feather(width, height)

        crop = Crop(left, right, top, bottom).pad(x_feather, y_feather)

        clamped = crop.clamp_to_parent(self.crop)

        if clamped.width() == 0 or clamped.height() == 0:
            return None
        else:
            return crop


    def bounds_mask(self, graph):
        if self.cached_bounds_mask is None:
            self.cached_bounds_mask = graph.node(
                "SolidMask",
                value=0.0,
                width=self.bounds.width(),
                height=self.bounds.height(),
            ).out(0)

        return self.cached_bounds_mask


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
        (x_feather, y_feather) = region.evaluate_feather(self.bounds.width(), self.bounds.height())

        # Crop which contains only solid white pixels
        solid_crop = crop.pad(-x_feather, -y_feather).clamp_to_parent(self.crop)

        # Region occupies the entire image, no need for masking
        if solid_crop == self.crop and region.strength == 1.0:
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

            if crop != self.bounds:
                # TODO figure out a way to not composite if the region is entirely within the snapped_crop
                mask = graph.node(
                    "MaskComposite",
                    destination=self.bounds_mask(),
                    source=mask,
                    x=crop.left,
                    y=crop.top,
                    operation="add",
                ).out(0)

            mask = self.apply_to_mask(graph, mask)

            return ProcessImage.apply_set_mask(graph, mask, region.strength, region.isolated, conditioning)
