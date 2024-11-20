"""DeepCompFedL: A Flower / PyTorch app."""

from deepcompfedl.compression.utils import get_params, set_params

import numpy as np
import torch


def prune(net, pruning_rate : float = 0.1):
    
    params = get_params(net)

    sorted = torch.cat([torch.from_numpy(i).flatten().abs() for i in params]).sort()[0]
    threshold = sorted[int(len(sorted)*pruning_rate)].item()
    
    for i,p in enumerate(params):
        params[i][np.abs(p)<=threshold] = 0

    set_params(net, params)

