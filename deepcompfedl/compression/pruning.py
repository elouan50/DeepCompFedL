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
    if 0. < pruning_rate < 1.:
        sorted = torch.cat([torch.from_numpy(i).flatten().abs() for i in params]).sort()[0]
        threshold = sorted[int(len(sorted) * pruning_rate)].item()
        
        for i, p in enumerate(params):
            prune_layer(params, i, p, threshold)
        
    return params

def prune_layer(params, i, layer, threshold):
    params[i][np.abs(layer) <= threshold] = 0