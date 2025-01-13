from deepcompfedl.models.resnet12 import ResNet12
import numpy as np
from pympler import asizeof

model = ResNet12(16, (3,32,32), 10)

total_size = 0
for param in model.parameters():
    total_size += np.prod(param.size())

print(f"The model has {total_size} total parameters.")
print(f"The memory size is of {asizeof.asizeof(model)} bytes.")
