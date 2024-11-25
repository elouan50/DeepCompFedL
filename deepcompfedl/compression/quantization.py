"""DeepCompFedL: A Flower / PyTorch app."""

from deepcompfedl.compression.utils import get_params, set_params

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def quantize(net, nbits : int = 8):
    """
    Applies weight sharing to the given model.
    Encompasses weights in 2**n clusters.
    Returns the representation of weights and a dictionnary.
    """
    for module in net.children():
        # We can't quantize a MaxPool layer, as it doesn't have weights
        if isinstance(module, nn.MaxPool2d):
            # print("MaxPool2d layer: no quantization")
            pass
        
        else:
            dev = module.weight.device
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            
            if isinstance(module, nn.Conv2d):
                # print(f"Conv2d layer   : {nbits} bits quantization")
                
                # Flattening the matrix
                flattened = np.zeros((shape[0]*shape[1]*shape[2], shape[3]))
                index = 0
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            flattened[index,:] = weight[i,j,k,:]
                            index += 1
                            
                # Using a compressed sparse representation for the matrix
                mat = csr_matrix(flattened) if shape[0]*shape[1]*shape[2] < shape[3] else csc_matrix(flattened)
                min_ = np.min(mat.data)
                max_ = np.max(mat.data)
                
                # Initializing the K-Means with a regular interval
                space = np.linspace(min_, max_, num=2**nbits)
                
                # Operating the K-Means
                kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
                kmeans.fit(mat.data.reshape(-1,1))
                new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                mat.data = new_weight
                new_flattened = mat.toarray()
                
                # Regiving the matrix of weights its original shape
                i = 0
                j = 0
                k = 0
                for k in range(new_flattened.shape[0]):
                    if j==shape[1]:
                        j = 0
                        i += 1
                    weight[i,j,:,:] = new_flattened[i*shape[0]+j*shape[1]:i*shape[0]+j*shape[1]+5, :]
                    k += 1
                
                # Updating the original model
                module.weight.data = torch.from_numpy(weight).to(dev)
            
            elif isinstance(module, nn.Linear):
                # print(f"Linear layer   : {nbits} bits quantization")
                            
                # Using a compressed sparse representation for the matrix
                mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
                min_ = np.min(mat.data)
                max_ = np.max(mat.data)
                
                # Initializing the K-Means with a regular interval
                space = np.linspace(min_, max_, 2**nbits)
                
                # Operating the K-Means
                kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
                kmeans.fit(mat.data.reshape(-1,1))
                new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                
                # Updating the original model
                mat.data = new_weight
                module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
            
            else:
                print(f"Unexpected layer: {type(module)}")
    
            
    

