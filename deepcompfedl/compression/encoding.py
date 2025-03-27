"""DeepCompFedL: A Flower / PyTorch app."""

import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

def huffman_encode(arr, prefix, save_dir='./', stats=False):
    """
    Encodes numpy array 'arr' and saves to `save_dir`
    The names of binary files are prefixed with `prefix`
    returns the number of bytes for the tree and the data after the compression
    """
    # Infer dtype
    dtype = str(arr.dtype)

    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32':float, 'int32':int, 'int64':int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while(len(heap) > 1):
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}

    def generate_code(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    generate_code(root, '')

    # Path to save location
    directory = Path(save_dir)

    # Dump data
    data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
    datasize = dump(data_encoding, directory/f'{prefix}.bin', stats)

    # Dump codebook (huffman tree)
    codebook_encoding = encode_huffman_tree(root, dtype)
    treesize = dump(codebook_encoding, directory/f'{prefix}_codebook.bin', stats)

    return treesize, datasize


# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    converter = {'float32':float2bitstr, 'int32':int2bitstr, 'int64':int2bitstr}
    code_list = []
    def encode_node(node):
        if node.value is not None: # node is leaf node
            code_list.append('1')
            lst = list(converter[dtype](node.value))
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)
    encode_node(root)
    return ''.join(code_list)


# My own dump / load logics
def dump(code_str, filename, stats=False):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    returns how many bytes are written
    """
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    if not(stats):
        with open(filename, 'wb') as f:
            f.write(byte_arr)
    return len(byte_arr)


# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]

def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr):
    return indptr[1:] - indptr[:-1]



# Encode / Decode models
def huffman_encode_model(model, directory='encodings/', stats=False):
    os.makedirs(directory, exist_ok=True)
    original_total = 0
    compressed_total = 0
    if stats:
        print(f"{'Layer':<15} | {'original':>10} {'compressed':>10} {'improvement':>11} {'percent':>7}")
        print('-'*70)
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            shape = weight.shape
            form = 'csc'
            mat = csc_matrix(weight.reshape(-1,1))

            # Encode
            t0, d0 = huffman_encode(mat.data, name+f'_{form}_data', directory, stats)
            t1, d1 = huffman_encode(mat.indices, name+f'_{form}_indices', directory, stats)

            # Print statistics
            original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
            compressed = t0 + t1 + t2 + d0 + d1 + d2

            if stats:
                print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        else: # bias
            # Note that we do not huffman encode bias
            bias = param.data.cpu().numpy()
            if not(stats):
                bias.dump(f'{directory}/{name}', stats)

            # Print statistics
            original = bias.nbytes
            compressed = original

            if stats:
                print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        original_total += original
        compressed_total += compressed

    if stats:
        print('-'*70)
        print(f"{'total':15} | {original_total:>10} {compressed_total:>10} {original_total / compressed_total:>10.2f}x {100 * compressed_total / original_total:>6.2f}%")
