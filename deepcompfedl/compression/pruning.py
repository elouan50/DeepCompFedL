""" DeepCompFedL: A Flower / PyTorch app.

This module provides functionality for pruning model parameters in a federated learning setting.
Functions:
    prune(params, pruning_rate: float = 0.1) -> list:
        Prunes the given parameters by setting values below a certain threshold to zero.
        Args:
            params (list): A list of numpy arrays representing the model parameters.
            pruning_rate (float): The fraction of parameters to prune. Default is 0.1.
        Returns:
            list: The pruned parameters.
"""

import numpy as np
import torch


def prune(params, pruning_rate : float = 0.1):
    if pruning_rate > 0.:
        sorted = torch.cat([torch.from_numpy(i).flatten().abs() for i in params]).sort()[0]
        threshold = sorted[int(len(sorted)*pruning_rate)].item()
        
        for i,p in enumerate(params):
            params[i][np.abs(p)<=threshold] = 0
        
    # set_weights(net, params)
    return params

