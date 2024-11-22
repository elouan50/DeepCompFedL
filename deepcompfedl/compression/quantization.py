"""DeepCompFedL: A Flower / PyTorch app."""

from deepcompfedl.compression.utils import get_params, set_params

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def quantize(net, nbits : int = 32):
    """
    Applies weight sharing to the given model.
    Encompasses weights in 2**n clusters.
    Returns the representation of weights and a dictionnary.
    """
    for module in net.children():
        # We can't quantize a MaxPool layer, as it doesn't have weights
        if isinstance(module, nn.MaxPool2d):
            print("MaxPool2d layer: no quantization")
        
        else:
            dev = module.weight.device
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            
            if isinstance(module, nn.Conv2d):
                print(f"Conv2d layer   : {nbits} bits quantization")
                print("Conv2d #TODO")
                # print(weight)
                
                # # Flattening the matrix
                # for i in range(shape[0]):
                #     for j in range(shape[1]):
                #         print(weight[i,j,:,:])
                        # mat = csr_matrix(weight[i,j,:,:]) if shape[0] < shape[1] else csc_matrix(weight[i,j,:,:])
                        # min_ = min(mat.data)
                        # max_ = max(mat.data)
                        # space = np.linspace(min_, max_, num=2**nbits)
                        # kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
                        # kmeans.fit(mat.data.reshape(-1,1))
                        # new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                        # mat.data = new_weight
                        # module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
            
            elif isinstance(module, nn.Linear):
                print(f"Linear layer   : {nbits} bits quantization")
                print("Linear #TODO")
                # mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
                # min_ = min(mat.data)
                # max_ = max(mat.data)
                # space = np.linspace(min_, max_, num=2**nbits)
                # kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
                # kmeans.fit(mat.data.reshape(-1,1))
                # new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                # mat.data = new_weight
                # module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
            
            else:
                print(f"Unexpected layer: {type(module)}")
    
            
    

