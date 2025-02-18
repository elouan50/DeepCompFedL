"""
This file aims to show how the measurement of the size
of the model in done in the project.
"""

import torch
import os

# from deepcompfedl.models.resnets import ResNet18
from deepcompfedl.models.resnet18 import ResNet18
from deepcompfedl.task import (
    train,
    test,
    load_data,
    get_weights,
    set_weights
)
from deepcompfedl.compression.pruning import prune
from deepcompfedl.compression.metrics import pruned_weights

save_dir = "deepcompfedl/saves/resnet18_grativol"

os.makedirs(save_dir, exist_ok=True)


def saveforprunerate(pr):
    # Initialize the model
    # model = ResNet18(64, (3,32,32), 10)
    model = ResNet18()

    # Train it on the first partition, print the accuracy
    trainloader, testloader = load_data(0, 10, 100, "CIFAR10")
    train(model, trainloader, 1, "cuda")
    print("     -----")
    print(f"Metrics before training: {test(model, testloader, 'cuda')}")

    # Prune the model
    params = get_weights(model)
    params = prune(params, pr)
    set_weights(model, params)
    pruned_weights(model)

    # Review accuracy after pruning
    print(f"Metrics after training: {test(model, testloader, 'cuda')}")
    
    torch.save(model, f"{save_dir}/pruned{pr}.ptmodel")

pruning_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

for pr in pruning_rates:
    saveforprunerate(pr)
