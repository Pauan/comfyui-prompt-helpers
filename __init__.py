from comfy_api.latest import ComfyExtension, io

from .src.prompt_helpers import (ez, prompt, replacements)


class PromptHelpers(ComfyExtension):
    async def on_load(self) -> None:
        await replacements.register()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ez.EZBatch,
            ez.EZBlank,
            ez.EZImage,
            ez.EZUpscaleTiled,
            ez.EZDetail,
            ez.EZCrop,
            ez.EZPrompt,
            ez.EZSampler,
            ez.EZFilename,
            ez.EZGenerate,
            ez.EZGenerateSave,
            ez.EZCheckpoint,
            ez.ConcatenateControlNet,
            ez.EmptyControlNet,
            ez.EZControlNet,
            ez.EZNotify,

            prompt.ProcessJson,
            prompt.ParseLines,
            prompt.ParseYAML,
            prompt.ConcatenateJson,
            prompt.PromptToggle,
            prompt.ApplyLoras,
            prompt.DebugJSON,
            prompt.DebugJSONPrompt,
        ]

async def comfy_entrypoint() -> PromptHelpers:
    return PromptHelpers()


WEB_DIRECTORY = "./js"

__all__ = [
    "comfy_entrypoint",
    "WEB_DIRECTORY",
]
