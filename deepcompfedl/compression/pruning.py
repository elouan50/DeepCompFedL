"""DeepCompFedL: A Flower / PyTorch app."""

from deepcompfedl.task import get_weights, set_weights

import numpy as np
import torch


def prune(params, pruning_rate : float = 0.1):

    sorted = torch.cat([torch.from_numpy(i).flatten().abs() for i in params]).sort()[0]
    threshold = sorted[int(len(sorted)*pruning_rate)].item()
    
    for i,p in enumerate(params):
        params[i][np.abs(p)<=threshold] = 0
    
    # set_weights(net, params)
    return params

