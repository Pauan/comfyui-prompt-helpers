from .src.prompt_helpers.nodes import (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

import os
import nodes
from comfy_config import config_parser

custom_node_dir = os.path.dirname(os.path.realpath(__file__))

project_config = config_parser.extract_node_configuration(custom_node_dir)

js_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

nodes.EXTENSION_WEB_DIRS[project_config.project.name] = js_dir

__author__ = """Pauan"""
__email__ = "pauanyu+github@pm.me"
__version__ = project_config.project.version

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
