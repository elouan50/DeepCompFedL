"""
This file aims to estimate the size of a model, by exploring
its attributes. This is not how I'll do it later on.

Actual method used for measuring size the model is described
in the `saveandzip.py` file.
"""

from deepcompfedl.models.resnet12 import ResNet12
from deepcompfedl.models.resnet18 import ResNet18
# from deepcompfedl.models.resnets import ResNet18
from deepcompfedl.models.qresnets import QResNet18
from deepcompfedl.task import train, test, load_data
import numpy as np
from pympler import asizeof

from torchvision.models import resnet18


# model = ResNet12(16, (3,32,32), 10)
# model = ResNet12(64, (3,32,32), 10)
# model = ResNet18(64, (3,32,32), 10)
model = ResNet18()
# model = QResNet18(64, (3,32,32), 10)
# model = resnet18(num_classes=10)

# trainloader, testloader = load_data(0,10, 100, "CIFAR10")
# train(model, trainloader, 1, "cuda")
# print(test(model, testloader, "cuda"))

total_size = 0
mem = 0
for param in model.parameters():
    total_size += np.prod(param.size())
    mem += asizeof.asizeof(param)

print(f"The model has {total_size} total parameters.")
print(f"The memory size is of {asizeof.asizeof(model)} or {mem} bytes.")
