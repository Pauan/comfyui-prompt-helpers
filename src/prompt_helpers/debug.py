def debug_image(graph, image, prefix="image"):
    graph.node("SaveImage", images=image, filename_prefix="DEBUG/" + prefix)
    return image


def debug_mask(graph, mask, prefix="mask"):
    image = graph.node("MaskToImage", mask=mask).out(0)

    graph.node("SaveImage", images=image, filename_prefix="DEBUG/" + prefix)
    return mask
