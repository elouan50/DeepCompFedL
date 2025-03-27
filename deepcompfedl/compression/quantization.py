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

def quantize(params: NDArrays,
             nbits: int = 8,
             layer_wise: bool = True,
             init_space: str = "uniform"
             ) -> NDArrays:
    """
    Quantize the weights of a model using K-Means clustering.

    Args:
        params (NDArrays): Parameters of the model.
        nbits (int, optional): Number of bits for cluster representation. Defaults to 8.
        layer_wise (bool, optional): If the quantization is applied to layers spearately (True) or all weights at the same time (False). Defaults to True.
        init_space (str, optional): The initialization space for the K-Means clustering. Defaults to "uniform". Can take values "uniform" or "random" or "density".

    Returns:
        NDArrays: Parameters of the quantized model.
    """
    
    if layer_wise: # Layer-wise quantization
        for i in range(len(params)):
            quantize_layer(params, i, nbits, init_space)
        
    else: # Global scale
        shape = []
        flattened = []
        for layer in params:
            shape.append(np.shape(layer))
            flattened.append(layer.flatten())
        
        flattened = np.concatenate(flattened)
        
        # Applying K-Means clustering
        flattened = apply_kmeans(flattened, nbits, init_space)

        flattened = np.split(flattened.reshape(-1,1), np.cumsum([np.prod(s) for s in shape])[:-1])
        for i in range(len(shape)):
            params[i] = flattened[i].reshape(shape[i])
        
    return params

def quantize_layer(params, i, nbits=8, init_space="uniform"):
    # Flatten the matrix
    layer = params[i]
    shape = np.shape(layer)            
    flattened = layer.reshape(-1,1)
    
    # Applying K-Means clustering
    flattened = apply_kmeans(flattened, nbits, init_space)
        
    params[i] = flattened.reshape(shape)
    

def apply_kmeans(mat, nbits=8, init_space="uniform"):
    """
    Apply K-Means clustering to the flattened matrix.
    
    Args:
        mat (np.ndarray): Flattened matrix.
        nbits (int): Number of bits for cluster representation.
        init_space (str): The initialization space for the K-Means clustering.
        
    Returns:
        np.ndarray: The quantized matrix.
    """
    # Using a compressed sparse representation for the flattened matrix
    if not isinstance(mat, csr_matrix):
        mat = csr_matrix(mat)
    
    if mat.getnnz() > 2**nbits:
        min_ = np.min(mat.data)
        max_ = np.max(mat.data)
        
        if min_ != max_:
            if init_space == "uniform":
                # Initializing the K-Means with a regular interval
                space = np.linspace(min_, max_, num=2**nbits)

            elif init_space == "random":
                # Initializing the K-Means with random values
                space = np.random.uniform(min_, max_, size=2**nbits)
            
            elif init_space == "density":
                # Initializing the K-Means with the most dense values
                hist, bins = np.histogram(mat.data, bins=2**nbits)
                space = bins[np.argmax(hist)+1:-1]
                
            # Operating the K-Means
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
            kmeans.fit(mat.data.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            flat = mat.toarray()
    
    return flat
