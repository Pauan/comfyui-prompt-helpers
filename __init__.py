from comfy_api.latest import ComfyExtension, io

from .src.prompt_helpers import (ez, prompt, replacements)


class PromptHelpers(ComfyExtension):
    async def on_load(self) -> None:
        await replacements.register()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            # controlnet
            ez.EZControlNet,
            ez.ConcatenateControlNet,
            ez.EmptyControlNet,

            # image
            ez.EZBlank,
            ez.EZImage,
            ez.EZUpscaleTiled,
            ez.EZBatch,
            ez.EZCrop,
            ez.EZDetail,

            # prompt
            ez.EZPrompt,
            ez.EZMaskRegions,
            prompt.PromptToggle,
            prompt.ParseLines,
            prompt.ParseYAML,
            prompt.ConcatenateJson,
            prompt.DebugJSON,
            prompt.DebugJSONPrompt,

            # sampler
            ez.EZSampler,

            # default
            ez.EZCheckpoint,
            ez.EZFilename,
            ez.EZGenerate,
            ez.EZGenerateSave,
            ez.EZNotify,

            # mask
            ez.MaskToBounds,

            # hidden
            prompt.ApplyLoras,
        ]

async def comfy_entrypoint() -> PromptHelpers:
    return PromptHelpers()


WEB_DIRECTORY = "./js"

__all__ = [
    "comfy_entrypoint",
    "WEB_DIRECTORY",
]
