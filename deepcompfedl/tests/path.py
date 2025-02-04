import numpy as np
import torch
import torch.nn as nn
import math
import time

import pickle

from torchvision.models import resnet18

from deepcompfedl.models.net import Net
from deepcompfedl.models.resnet12 import ResNet12
from deepcompfedl.models.resnet18 import ResNet18, ResNet, ResidualBlock
# from deepcompfedl.models.resnets import ResNet18, ResNet, BasicBlock
from deepcompfedl.task import train, load_data

from deepcompfedl.compression.pruning import prune

net = Net()
# resNet18 = resnet18()
# resNet18 = ResNet18(64, (3,32,32), 10, True)
resNet18 = ResNet18()
resNet12 = ResNet12(16, (3,32,32), 10)

compare = [[],[],[]]



##### With parameters()

begin = time.perf_counter()

params = [param for param in resNet18.parameters()]

i = 0
for param in params:
    size = tuple(param.size())
    temp = 1
    for j in range(len(size)):
        temp *= size[j]
    i += temp
    compare[0].append(size)

end = time.perf_counter()

print(f"For ResNet18, {len(params)} layers ({i} params) in {int((end-begin)*1000000)}ms for parameters()")

##### With state_dict().items()

begin = time.perf_counter()

params = [val.cpu().numpy() for _, val in resNet18.state_dict().items()]

j = 0
for param in params:
    j += np.size(param)
    compare[1].append(np.shape(param))

params = prune(params)

end = time.perf_counter()

# print(f"For ResNet18, {len(params)} layers ({j} params) in {int((end-begin)*1000000)}ms for state_dict().items()")


##### With children()

begin = time.perf_counter()

def print_layers_resnet(resnet: ResNet, k=0):
    for module in resnet.children():
        if isinstance(module, (nn.Sequential, ResidualBlock)):
            print(f"info:        {type(module)}")
            k = print_layers_resnet(module, k)
        else:
            print(type(module))
            k += 1
    return k

k = print_layers_resnet(resNet18)

end = time.perf_counter()

print(f"For recursive children exploration, {k} layers in {int((end-begin)*1000000)}ms.")

##### Print the results

print(compare[0][:]==compare[1][:])
