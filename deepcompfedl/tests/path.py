import numpy as np
import torch
import math
import time

import pickle

from torchvision.models import resnet18

from deepcompfedl.models.net import Net
from deepcompfedl.models.resnet12 import ResNet12
from deepcompfedl.task import train, load_data

from deepcompfedl.compression.pruning import prune

net = Net()
resNet18 = resnet18()
resNet12 = ResNet12(16, (3,32,32), 10)

compare = [[],[]]

### With parameters()

begin1 = time.perf_counter()

params = [param for param in resNet18.parameters()]

i = 0
for param in params:
    size = tuple(param.size())
    temp = 1
    for j in range(len(size)):
        temp *= size[j]
    i += temp
    compare[0].append(size)

# params = prune(params)

end1 = time.perf_counter()

### With state_dict().items()

begin2 = time.perf_counter()

palams = [val.cpu().numpy() for _, val in resNet18.state_dict().items()]

j = 0
for palam in palams:
    j += np.size(palam)
    compare[1].append(np.shape(palam))

palams = prune(palams)

end2 = time.perf_counter()

### Print the results

print(f"For ResNet18, {len(params)} layers ({i} params) in {int((end1-begin1)*1000000)}ms or {len(palams)} layers ({j} params) in {int((end2-begin2)*1000000)}ms")
print(len(compare[0]))
print(compare[0][:]==compare[1][:])
print(len(compare[1]))
print(param[size[0]-1])
print(palam[size[0]-1])