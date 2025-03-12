"""DeepCompFedL: A Flower / PyTorch app.

This module defines the QResNet12 model and its basic building block, QBasicBlockRN12.
Classes:
    QBasicBlockRN12(nn.Module): A basic block for ResNet12 consisting of three convolutional layers with GroupNorm and Leaky ReLU activations.
    QResNet12(nn.Module): A ResNet12 model composed of multiple BasicBlockRN12 blocks, followed by a fully connected layer for classification.
QBasicBlockRN12:
    Methods:
        __init__(self, in_planes, planes): Initializes the BasicBlockRN12 with the given input and output planes.
        forward(self, x): Defines the forward pass of the BasicBlockRN12.
QResNet12:
    Methods:
        __init__(self, feature_maps: int = 16, input_shape: tuple = (3,32,32), num_classes: int = 10): Initializes the ResNet12 model with the given feature maps, input shape, and number of classes.
        forward(self, x): Defines the forward pass of the ResNet12 model.
"""

import torch.nn as nn
import torch.nn.functional as F
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

class QBasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes, weight_quant):
        super(QBasicBlockRN12, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_planes, planes, kernel_size=3, padding=1, bias=False, weight_quant=weight_quant)
        self.bn1 = nn.GroupNorm(2, planes)        
        self.conv2 = qnn.QuantConv2d(planes, planes, kernel_size=3, padding=1, bias=False, weight_quant=weight_quant)
        self.bn2 = nn.GroupNorm(2, planes)
        self.conv3 = qnn.QuantConv2d(planes, planes, kernel_size=3, padding=1, bias=False, weight_quant=weight_quant)
        self.bn3 = nn.GroupNorm(2, planes)
        
        norm = nn.GroupNorm(2,planes)

        self.shortcut = nn.Sequential(
            qnn.QuantConv2d(in_planes, planes, kernel_size=1, bias=False, weight_quant=weight_quant),norm)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return out
    
class QResNet12(nn.Module):
    def __init__(self, feature_maps: int = 16, input_shape: tuple = (3,32,32), num_classes: int = 10, weight_quant: WeightQuantType | int | None = Int8WeightPerTensorFloat):
        super(QResNet12, self).__init__()        
        if type(weight_quant) == int:
            if weight_quant == 8:
                weight_quant = Int8WeightPerTensorFloat
            elif weight_quant == 4:
                weight_quant = Int4WeightPerTensorFloatDecoupled
            else:
                weight_quant = NoneWeightQuant
        layers = []
        layers.append(QBasicBlockRN12(input_shape[0], feature_maps, weight_quant=weight_quant))
        layers.append(QBasicBlockRN12(feature_maps, int(2.5 * feature_maps), weight_quant=weight_quant))
        layers.append(QBasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps, weight_quant=weight_quant))
        layers.append(QBasicBlockRN12(5 * feature_maps, 10 * feature_maps, weight_quant=weight_quant))        
        self.layers = nn.Sequential(*layers)
        self.linear = qnn.QuantLinear(10 * feature_maps, num_classes, weight_quant=weight_quant)
        
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        return out
