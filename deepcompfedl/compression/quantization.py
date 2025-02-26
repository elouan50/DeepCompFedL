"""DeepCompFedL: A Flower / PyTorch app.

This module provides functionality for quantizing the weights of a model using K-Means clustering.
Functions:
    quantize(params: NDArrays, nbits: int = 8, layer_scale: bool = True) -> NDArrays
            layer_scale (bool, optional): If the quantization is applied to layers separately (True) or all weights at the same time (False). Defaults to True.

"""

import numpy as np
from cuml.cluster import KMeans
from scipy.sparse import csr_matrix

from flwr.common import NDArrays

def quantize(params: NDArrays, nbits: int = 8, layer_scale: bool = True):
    """
    Quantize the weights of a model using K-Means clustering.

    Args:
        params (NDArrays): Parameters of the model.
        nbits (int, optional): Number of bits for cluster representation. Defaults to 8.
        layer_scale (bool, optional): If the quantization is applied to layers spearately (True) or all weights at the same time (False). Defaults to True.

    Returns:
        NDArrays: Parameters of the quantized model.
    """
    output = []
    
    if layer_scale: # Layer-wise quantization
        for layer in params:
            # Flatten the matrix
            shape = np.shape(layer)            
            flattened = layer.reshape(-1,1)
            
            # Using a compressed sparse representation for the flattened matrix
            mat = csr_matrix(flattened)
            if mat.getnnz() > 2**nbits:
                min_ = np.min(mat.data)
                max_ = np.max(mat.data)
                
                if min_ != max_:    
                    # Initializing the K-Means with a regular interval
                    space = np.linspace(min_, max_, num=2**nbits)
                        
                    # Operating the K-Means
                    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
                    kmeans.fit(mat.data.reshape(-1,1))
                    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                    mat.data = new_weight
                    flattened = mat.toarray()
                
            output.append(flattened.reshape(shape))
        
    else: # Global scale
        shape = []
        flattened = []
        for layer in params:
            shape.append(np.shape(layer))
            flattened.append(layer.flatten())
        
        flattened = np.concatenate(flattened)
        
        # Using a compressed sparse representation for the flattened matrix
        mat = csr_matrix(flattened)
        if mat.getnnz() > 2**nbits:
            min_ = np.min(mat.data)
            max_ = np.max(mat.data)
            
            if min_ != max_:    
                # Initializing the K-Means with a regular interval
                space = np.linspace(min_, max_, num=2**nbits)
                    
                # Operating the K-Means
                kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
                kmeans.fit(mat.data.reshape(-1,1))
                new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                mat.data = new_weight
                flattened = mat.toarray()
                
        flattened = np.split(flattened.reshape(-1,1), np.cumsum([np.prod(s) for s in shape])[:-1])
        for i in range(len(shape)):
            output.append(flattened[i].reshape(shape[i]))
        
    return output
