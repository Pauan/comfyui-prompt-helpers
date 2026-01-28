from comfy_api.latest import ComfyExtension, io

from .src.prompt_helpers import (ez, prompt)


class PromptHelpers(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ez.EZBatch,
            ez.EZBlank,
            ez.EZImage,
            ez.EZInpaint,
            ez.EZPrompt,
            ez.EZSampler,
            ez.EZFilename,
            ez.EZGenerate,

            prompt.ParseLines,
            prompt.ParseYAML,
            prompt.ConcatenateJson,
            prompt.FromJSON,
            prompt.PromptToggle,
        ]

async def comfy_entrypoint() -> PromptHelpers:
    return PromptHelpers()


WEB_DIRECTORY = "./js"

__all__ = [
    "comfy_entrypoint",
    "WEB_DIRECTORY",
]
