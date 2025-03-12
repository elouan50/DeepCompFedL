"""DeepCompFedL: A Flower / PyTorch app.

This implementation of the QResNet18 is adapted from the ResNet18 implementation
found at deepcompfedl/models/resnet18.py.

Classes:
    QResidualBlock: Defines a residual block used in QResNet18.
    QResNet: Defines the QResNet18 architecture.
    
Functions:
    QResNet18: Returns an instance of the QResNet18 model.
"""

from torch import nn
from torch.nn import functional as F
import brevitas.nn as qnn

from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol
from typing import Type, Union

from brevitas.quant import (
    NoneWeightQuant,
    Int8WeightPerTensorFloat,
    Int4WeightPerTensorFloatDecoupled,
    )

WeightQuantType = Union[WeightQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]

class QResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, weight_quant=Int8WeightPerTensorFloat):
        super(QResidualBlock, self).__init__()
        self.left = nn.Sequential(
            qnn.QuantConv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False, weight_quant=weight_quant),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            qnn.QuantConv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False, weight_quant=weight_quant),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False, weight_quant=weight_quant),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class QResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, weight_quant=Int8WeightPerTensorFloat):
        super(QResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            qnn.QuantConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, weight_quant=weight_quant),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_qlayer(ResidualBlock, 64, 2, stride=1, weight_quant=weight_quant)
        self.layer2 = self.make_qlayer(ResidualBlock, 128, 2, stride=2, weight_quant=weight_quant)
        self.layer3 = self.make_qlayer(ResidualBlock, 256, 2, stride=2, weight_quant=weight_quant)        
        self.layer4 = self.make_qlayer(ResidualBlock, 512, 2, stride=2, weight_quant=weight_quant)        
        self.fc = qnn.QuantLinear(512, num_classes, weight_quant=weight_quant)
        
    def make_qlayer(self, block, channels, num_blocks, stride, weight_quant):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, weight_quant=weight_quant))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def QResNet18(num_classes: int = 10, weight_quant: WeightQuantType | int | None = Int8WeightPerTensorFloat) -> QResNet:
    if type(weight_quant) == int:
        if weight_quant == 8:
            wquant = Int8WeightPerTensorFloat
        elif weight_quant == 4:
            wquant = Int4WeightPerTensorFloatDecoupled
        else:
            wquant = NoneWeightQuant
    return QResNet(QResidualBlock, num_classes=num_classes, weight_quant=wquant)

