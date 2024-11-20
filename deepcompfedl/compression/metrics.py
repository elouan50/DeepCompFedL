"""DeepCompFedL: A Flower / PyTorch app."""

from deepcompfedl.compression.utils import get_params

import numpy as np


def pruned_weights(net):
    count_zeros = 0
    count_total = 0
    
    params = get_params(net)
    for i,p in enumerate(params):
        total = np.size(p)
        zeros = total - np.count_nonzero(p)
        
        count_total += total
        count_zeros += zeros
        
        print(f"For layer {i}, {round(zeros/total*100,2)}% pruning (over {total} parameters).")
    
    print(f"Global pruning: {round(count_zeros/count_total*100,2)}%")
    
        