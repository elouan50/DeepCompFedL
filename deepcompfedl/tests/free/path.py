"""
This file aims to test and understand different path explorations
for the neural network we use.
Feel free to change the net, and evaluate different metrics.
"""

import numpy as np
import torch.nn as nn
import time

from deepcompfedl.models.net import Net
from deepcompfedl.models.resnet12 import ResNet12
from deepcompfedl.models.resnet18 import ResNet18, ResNet, ResidualBlock


# net = Net()
# net = ResNet12()
net = ResNet18()


##### With parameters()

begin = time.perf_counter()

params = [param for param in net.parameters()]

i = 0
for param in params:
    size = tuple(param.size())
    temp = 1
    for j in range(len(size)):
        temp *= size[j]
    i += temp

end = time.perf_counter()

print(f"For parameters() exploration, {len(params)} layers ({i} params) in {int((end-begin)*1000000)}ms.")


##### With state_dict().items()

begin = time.perf_counter()

params = [val.cpu().numpy() for _, val in net.state_dict().items()]

j = 0
for param in params:
    j += np.size(param)

end = time.perf_counter()

print(f"For state_dict().items() exploration, {len(params)} layers ({j} params) in {int((end-begin)*1000000)}ms.")


##### With children()

begin = time.perf_counter()

def print_layers_resnet(resnet: ResNet, k=0):
    for module in resnet.children():
        if isinstance(module, (nn.Sequential, ResidualBlock)):
            # print(f"info:        {type(module)}")
            k = print_layers_resnet(module, k)
        else:
            # print(type(module))
            k += 1
    return k

k = print_layers_resnet(net)

end = time.perf_counter()

print(f"For recursive children() exploration, {k} layers in {int((end-begin)*1000000)}ms.")


##### With modules()

begin = time.perf_counter()

def print_modules_resnet(resnet: ResNet, l=0):
    for module in resnet.modules():
        if isinstance(module, (nn.Sequential, ResidualBlock)):
            # print(f"info:        {type(module)}")
            pass
        else:
            # print(type(module))
            l += 1
    return l

l = print_modules_resnet(net)

end = time.perf_counter()

print(f"For modules() exploration, {l} layers in {int((end-begin)*1000000)}ms.")
