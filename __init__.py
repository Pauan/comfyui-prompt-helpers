from .src.prompt_helpers.nodes import (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)

import os
import nodes
from comfy_config import config_parser

__author__ = """Pauan"""
__email__ = "pauanyu+github@pm.me"
__version__ = "0.0.1"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

custom_node_dir = os.path.dirname(os.path.realpath(__file__))

project_config = config_parser.extract_node_configuration(custom_node_dir)

js_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

nodes.EXTENSION_WEB_DIRS[project_config.project.name] = js_dir
