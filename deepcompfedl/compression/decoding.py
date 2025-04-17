"""DeepCompFedL: A Flower / PyTorch app."""

import numpy as np
import cupy as cp
import torch

from logging import WARNING
from flwr.common.logger import log

from scipy.sparse import csr_matrix, csc_matrix
from cupyx.scipy.sparse import csc_matrix as csc_gpu

from collections import namedtuple
from pathlib import Path
import struct

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq


def huffman_decode_model(model, directory='encodings/'):
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'weight' in name or 'bias' in name:
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            weight = param.data.cpu().numpy()
            shape = weight.shape
            form = 'csc'
            matrix = csc_gpu

            # Decode data
            data = huffman_decode(directory, name+f'_{form}_data', dtype='float32')
            indices = cp.cumsum(huffman_decode(directory, name+f'_{form}_indices', dtype='int32'))
            indptr = cp.array([0, len(indices)])
            
            # Construct matrix
            if len(data) == 0:
                param.data = torch.zeros(shape).to(dev)
            else:
                if indices.size == 0:
                    indices = cp.array([i for i in range(len(data))])
                    indptr = cp.array([0, len(indices)], dtype='int32')
                    mat = matrix((data, indices, indptr)).todense().get()
                else:
                    mat = matrix((data, indices, indptr)).todense().get()

                # Insert to model
                # When the last weights of a layer in the initial are null,
                # the CSC representation doesn't represent them ;
                # we need to add them back
                forgot_weights = np.prod(shape) - len(mat)
                if forgot_weights == 0:
                    param.data = torch.from_numpy(mat).reshape(shape).to(dev)
                else:
                    param.data = torch.from_numpy(np.concatenate([mat.flatten(), np.zeros(forgot_weights)])).reshape(shape).to(dev)
        else:
            log(WARNING, "Parameter not recognized")
    return model

def reconstruct_indptr(diff):
    return np.concatenate([[0], np.cumsum(diff)])

def decode_huffman_tree(code_str, dtype):
    """
    Decodes a string of '0's and '1's and constructs a huffman tree
    """
    converter = {'float32':bitstr2float, 'int32':bitstr2int}
    idx = 0
    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1': # Leaf node
            value = converter[dtype](code_str[idx:idx+32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()

def huffman_decode(directory, prefix, dtype):
    """
    Decodes binary files from directory
    """
    directory = Path(directory)

    # Read the codebook
    codebook_encoding = load(directory/f'{prefix}_codebook.bin')
    if codebook_encoding == '':
        return cp.array([])
    root = decode_huffman_tree(codebook_encoding, dtype)

    # Read the data
    data_encoding = load(directory/f'{prefix}.bin')

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None: # Leaf node
            data.append(ptr.value)
            ptr = root

    return cp.array(data, dtype=dtype)


def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    try:
        with open(filename, 'rb') as f:
            header = f.read(1)
            rest = f.read() # bytes
            code_str = ''.join(f'{byte:08b}' for byte in rest)
            offset = ord(header)
            if offset != 0:
                code_str = code_str[:-offset] # string of '0's and '1's
    except FileNotFoundError:
        code_str = ''
    return code_str


# Helper functions for converting between bit string and (float or int)
def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]

def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]
