"""DeepCompFedL: A Flower / PyTorch app."""

from pympler import asizeof

from deepcompfedl.task import get_weights

import numpy as np
import torch.nn as nn


def pruned_weights(net):
    count_zeros = 0
    count_total = 0
    
    params = get_weights(net)
    for i,p in enumerate(params):
        total = np.size(p)
        zeros = total - np.count_nonzero(p)
        
        count_total += total
        count_zeros += zeros
        
        print(f"For layer {i}, {round(zeros/total*100,4)}% pruning (over {total} parameters).")
    
    print(f"Global pruning: {round(count_zeros/count_total*100,2)}%")

def quantized_model(net):
    list_weights = [0.]
    
    for module in net.children():
        if not(isinstance(module, nn.MaxPool2d)):
            weight = module.weight.data.cpu().numpy()
            size = np.size(weight)
            layer = np.reshape(weight, (size,1))
            for w in layer:
                if not(w in list_weights):
                    list_weights.append(w.item())
    
    print(f"There are {len(list_weights)-1} different weights different than 0 in the model.")
    print(f"These weights are: {list_weights[1:]}")

def quantized_layers(net):
    for module in net.children():
        if isinstance(module, nn.MaxPool2d):
            pass
        elif isinstance(module, nn.Sequential):
            pass
            # for basicblock in module.children():
            #     quantized_layers(basicblock)
        elif isinstance(module, nn.GroupNorm):
            pass
        else:
            print(f"{type(module)} is being examined")
            weight = module.weight.data.cpu().numpy()
            size = np.size(weight)
            layer = np.reshape(weight, (size,1))
            list_weights = [0.]
            for w in layer:
                if not(w in list_weights):
                    list_weights.append(w.item())

            print(f"    Layer {module.__class__.__name__}: there are {len(list_weights)-1} different weights different than 0.")
            # print(f"These weights are: {list_weights}")

def size_var(net):
    size = asizeof.asizeof(net)
    print(f"This net weights currently {size} bytes.")
    return size
