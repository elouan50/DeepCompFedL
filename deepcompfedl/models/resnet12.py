"""DeepCompFedL: A Flower / PyTorch app.

This module defines the ResNet12 model and its basic building block, BasicBlockRN12.
Classes:
    BasicBlockRN12(nn.Module): A basic block for ResNet12 consisting of three convolutional layers with GroupNorm and Leaky ReLU activations.
    ResNet12(nn.Module): A ResNet12 model composed of multiple BasicBlockRN12 blocks, followed by a fully connected layer for classification.
BasicBlockRN12:
    Methods:
        __init__(self, in_planes, planes): Initializes the BasicBlockRN12 with the given input and output planes.
        forward(self, x): Defines the forward pass of the BasicBlockRN12.
ResNet12:
    Methods:
        __init__(self, feature_maps: int = 16, input_shape: tuple = (3,32,32), num_classes: int = 10): Initializes the ResNet12 model with the given feature maps, input shape, and number of classes.
        forward(self, x): Defines the forward pass of the ResNet12 model.
"""

import torch.nn as nn
import torch.nn.functional as F

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes)        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.GroupNorm(2, planes)
        
        norm = nn.GroupNorm(2,planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),norm)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps: int = 16, input_shape: tuple = (3,32,32), num_classes: int = 10):
        super(ResNet12, self).__init__()        
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))        
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(10 * feature_maps, num_classes)
        
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
