"""
Unit tests for deepcompfedl models.

This module contains unit tests for the following models:
- Net
- ResNet12
- ResNet18

Each test verifies that the model can process a dummy input tensor and produce an output tensor with the expected shape.
"""

import torch
import pytest
from deepcompfedl.models.net import Net
from deepcompfedl.models.resnet12 import ResNet12
from deepcompfedl.models.resnet18 import ResNet18

@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 32, 32)  # Example input tensor

def test_net(dummy_input):
    model = Net()
    output = model(dummy_input)
    assert output is not None
    assert output.shape[0] == dummy_input.shape[0]

def test_resnet12(dummy_input):
    model = ResNet12()
    output = model(dummy_input)
    assert output is not None
    assert output.shape[0] == dummy_input.shape[0]

def test_resnet18(dummy_input):
    model = ResNet18()
    output = model(dummy_input)
    assert output is not None
    assert output.shape[0] == dummy_input.shape[0]
