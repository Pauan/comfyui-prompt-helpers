from .ComfyUIFEExampleVueBasic import NODE_CLASS_MAPPINGS
import os
import nodes
from comfy_config import config_parser

custom_node_dir = os.path.dirname(os.path.realpath(__file__))
print("==========================")

project_config = config_parser.extract_node_configuration(custom_node_dir)

print(project_config.project.name)

print("==========================")

js_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

nodes.EXTENSION_WEB_DIRS[project_config.project.name] = js_dir

__all__ = ['NODE_CLASS_MAPPINGS']


"""Top-level package for myfootest2."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",

]

__author__ = """Pauan"""
__email__ = "pauanyu+github@pm.me"
__version__ = "0.0.1"

from .src.myfootest2.nodes import NODE_CLASS_MAPPINGS
from .src.myfootest2.nodes import NODE_DISPLAY_NAME_MAPPINGS


