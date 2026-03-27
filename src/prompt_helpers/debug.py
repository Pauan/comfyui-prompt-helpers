def debug_image(graph, image):
    graph.node("SaveImage", images=image, filename_prefix="DEBUG/image")


def debug_mask(graph, mask):
    image = graph.node("MaskToImage", mask=mask).out(0)

    graph.node("SaveImage", images=image, filename_prefix="DEBUG/mask")
