"""
This file aims to test the models implemented in the project.
"""
import torch
import pytest
from deepcompfedl.models.net import Net
from deepcompfedl.models.qresnets import QResNet18, QResNet8, QResNet20
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

def test_qresnet18(dummy_input):
    model = QResNet18()
    output = model(dummy_input)
    assert output is not None
    assert output.shape[0] == dummy_input.shape[0]

def test_qresnet8(dummy_input):
    model = QResNet8()
    output = model(dummy_input)
    assert output is not None
    assert output.shape[0] == dummy_input.shape[0]

def test_qresnet20(dummy_input):
    model = QResNet20()
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
