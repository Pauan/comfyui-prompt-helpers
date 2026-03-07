from comfy_api.latest import (io, ComfyAPI)

api = ComfyAPI()

async def register():
    await api.node_replacement.register(io.NodeReplace(
        old_node_id="prompt_helpers: EZInpaint",
        new_node_id="prompt_helpers: EZImage",
        old_widget_ids=["image", "mask", "image_weight", "grow_mask"],
        input_mapping=[
            {"old_id": "image", "new_id": "image"},
            {"old_id": "mask", "new_id": "mask"},
            {"old_id": "image_weight", "new_id": "image_weight"},
        ],
    ))
