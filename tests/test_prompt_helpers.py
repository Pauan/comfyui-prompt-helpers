#!/usr/bin/env python

"""Tests for `Prompt Helpers` package."""

import pytest
from src.prompt_helpers.nodes import PromptToggle

@pytest.fixture
def example_node():
    """Fixture to create an Example node instance."""
    return PromptToggle()

def test_example_node_initialization(example_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_node, PromptToggle)

def test_return_types():
    """Test the node's metadata."""
    assert PromptToggle.RETURN_TYPES == ("CONDITIONING",)
    assert PromptToggle.FUNCTION == "encode"
    assert PromptToggle.CATEGORY == "conditioning/helpers"
