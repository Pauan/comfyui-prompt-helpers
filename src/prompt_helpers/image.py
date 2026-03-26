def clamp(value, low, high):
    return max(low, min(value, high))


class Crop:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


    def clamp(self, image_width, image_height):
        x = clamp(self.x, 0, image_width)
        y = clamp(self.y, 0, image_height)

        width = clamp(self.width, 0, image_width - x)
        height = clamp(self.height, 0, image_height - y)

        return Crop(x, y, width, height)


    def apply_to_image(self, graph, image):
        return graph.node("ImageCrop", image=image, x=self.x, y=self.y, width=self.width, height=self.height).out(0)


class Detail:
    def __init__(self, resize_multiplier, scale_method):
        self.resize_multiplier = resize_multiplier
        self.scale_method = scale_method


    def invert(self):
        resize_multiplier = 1.0 / self.resize_multiplier

        # We use `area` algorithm for downscaling.
        if resize_multiplier < 1.0:
            scale_method = "area"
        else:
            scale_method = self.scale_method

        return Detail(resize_multiplier, scale_method)


    def apply_to_image(self, graph, image):
        if self.resize_multiplier == 1.0:
            return image

        else:
            return graph.node(
                "ImageScaleBy",
                image=image,
                scale_by=self.resize_multiplier,
                upscale_method=self.scale_method,
            ).out(0)


    # TODO replace with ResizeImageMaskNode after https://github.com/Comfy-Org/ComfyUI/issues/12566 is fixed
    def apply_to_mask(self, graph, mask):
        if self.resize_multiplier == 1.0:
            return mask

        else:
            image = graph.node("MaskToImage", mask=mask).out(0)

            resized = graph.node(
                "ImageScaleBy",
                image=image,
                scale_by=self.resize_multiplier,
                upscale_method=self.scale_method,
            ).out(0)

            return graph.node("ImageToMask", image=resized, channel="red").out(0)


class ProcessImage:
    def __init__(self, crop, detail, width, height, batch_size, select_index):
        if crop is not None:
            self.crop = crop.clamp(width, height)
        else:
            self.crop = None

        self.detail = detail
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.select_index = select_index


    def empty_latent(self, graph):
        width = self.width
        height = self.height

        if self.crop is not None:
            width = self.crop.width
            height = self.crop.height

        if self.detail is not None:
            # https://github.com/Comfy-Org/ComfyUI/blob/602b2505a4ffeff4a732b8727ce27d3c2a1ef752/comfy_extras/nodes_post_processing.py#L284-L285
            width = int(round(width * self.detail.resize_multiplier))
            height = int(round(height * self.detail.resize_multiplier))

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
        if self.detail is not None:
            image = self.detail.invert().apply_to_image(graph, image)

        return image


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
            mask = graph.node("CropMask", mask=mask, x=self.crop.x, y=self.crop.y, width=self.crop.width, height=self.crop.height).out(0)

        return mask


    def resize_mask(self, graph, mask):
        if mask is not None and self.detail is not None:
            mask = self.detail.apply_to_mask(mask)

        return mask


    def composite_image(self, graph, original_image, downscaled_image, cropped_mask):
        if cropped_mask is None and self.crop is None:
            return downscaled_image

        else:
            x = 0
            y = 0

            if self.crop:
                x = self.crop.x
                y = self.crop.y

            return graph.node(
                "ImageCompositeMasked",
                destination=self.repeat_image(graph, original_image),
                source=downscaled_image,
                mask=cropped_mask,
                x=x,
                y=y,
                resize_source=False,
            ).out(0)
