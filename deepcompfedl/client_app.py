"""DeepCompFedL: A Flower / PyTorch app."""

import time
import os
import shutil
import torch
from logging import WARNING
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from flwr.common.logger import log

from deepcompfedl.compression.pruning import prune, prune_layer
from deepcompfedl.compression.quantization import quantize, quantize_layer
from deepcompfedl.compression.encoding import huffman_encode, huffman_encode_model
from deepcompfedl.compression.metrics import (
    pruned_weights,
    quantized,
)

from deepcompfedl.task import (
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)
from deepcompfedl.models.net import Net
from deepcompfedl.models.resnet12 import ResNet12
from deepcompfedl.models.resnet18 import ResNet18
from deepcompfedl.models.qresnet12 import QResNet12
from deepcompfedl.models.qresnet18 import QResNet18

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csc_matrix as csc_gpu
from cuml.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self,
                 net,
                 model_name,
                 trainloader,
                 valloader,
                 learning_rate,
                 local_epochs,
                 enable_pruning,
                 pruning_rate,
                 enable_quantization,
                 bits_quantization,
                 partition_id,
                 layer_compression,
                 init_space_quantization,
                 full_compression,
                 ): 
        self.net = net
        self.model_name = model_name
        self.trainloader = trainloader
        self.valloader = valloader
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.enable_pruning = enable_pruning
        self.pruning_rate = pruning_rate
        self.enable_quantization = enable_quantization
        self.bits_quantization = bits_quantization
        self.partition_id = partition_id
        self.layer_compression = layer_compression
        self.init_space_quantization = init_space_quantization
        self.full_compression = full_compression
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        begin = time.perf_counter()
        
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.learning_rate,
            self.local_epochs,
            self.device,
        )
        
        after_training = time.perf_counter()
        train_time = after_training - begin
        
        if self.full_compression:
            diff=[]
            centroids=[]
            indices=[]
            
            if os.path.exists(f'deepcompfedl/encodings/cl{self.partition_id}'):
                shutil.rmtree(f'deepcompfedl/encodings/cl{self.partition_id}')
            os.makedirs(f'deepcompfedl/encodings/cl{self.partition_id}', exist_ok=True)
            
            for name, param in self.net.named_parameters():
                ### Apply Pruning
                if  0 < self.pruning_rate < 1:
                    sorted = torch.cat([param.flatten().abs()]).sort()[0]
                    threshold = sorted[int(len(sorted) * self.pruning_rate)].item()
                    param.data[param.abs() <= threshold] = 0
                
                ### Apply Quantization
                form = 'csc'
                compressed = csc_gpu(cp.array(param.data.reshape(-1,1)))
                
                if self.bits_quantization < 32 and compressed.getnnz() > 2**self.bits_quantization:
                    min_, max_ = compressed.min(), compressed.max()
                    if min_ != max_:
                        space = np.linspace(min_, max_, num=2**self.bits_quantization)
                        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1)
                        kmeans.fit(compressed.data.reshape(-1,1))
                        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                        compressed.data = new_weight
                        param.data = torch.as_tensor(compressed.toarray()).reshape(param.data.shape)

                if compressed.data.size > 0:
                    ### Huffman encode
                    if 'weight' in name or 'bias' in name:
                        # For index difference
                        t0, d0 = huffman_encode(compressed.data.get(), name+f'_{form}_data', f'deepcompfedl/encodings/cl{self.partition_id}')
                        # For centroids
                        t1, d1 = huffman_encode(compressed.indices.get(), name+f'_{form}_indices', f'deepcompfedl/encodings/cl{self.partition_id}')
                        # For indices
                    else:
                        log(WARNING, "Parameter not recognized")
                                
            return [self.partition_id], len(self.trainloader.dataset), {"train-loss": train_loss, "training-time": train_time, "compression-time": time.perf_counter() - after_training}
                
        else:
            params = get_weights(self.net)
            if self.layer_compression:
                for i, layer in enumerate(params):
                    ### Apply Pruning
                    if self.enable_pruning:
                        sorted = torch.cat([torch.from_numpy(layer).flatten().abs()]).sort()[0]
                        threshold = sorted[int(len(sorted) * self.pruning_rate)].item()
                        prune_layer(params, i, threshold)
                    
                    ### Apply Quantization
                    if self.enable_quantization:
                        quantize_layer(params, i, self.bits_quantization, self.init_space_quantization)
            
            else:
                ### Apply Pruning
                if self.enable_pruning:
                    params = prune(params, self.pruning_rate)
                
                ### Apply Quantization
                if self.enable_quantization and self.model_name[0] != "Q":
                    params = quantize(params,
                                    self.bits_quantization,
                                    self.layer_compression,
                                    self.init_space_quantization)
            
            return params, len(self.trainloader.dataset), {"train-loss": train_loss, "training-time": train_time, "compression-time": time.perf_counter() - after_training}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        ### Apply Pruning
        if self.enable_pruning:
            ## Print stats for pruning
            if self.partition_id == 0:
                print(f"Effective received pruning (for client {self.partition_id} in client_fn):")
                pruned_weights(self.net)
                
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    ### Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    dataset = context.run_config["dataset"]
    alpha = context.run_config["alpha"]
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, alpha, dataset, batch_size)
    learning_rate = context.run_config["learning-rate"]
    local_epochs = context.run_config["client-epochs"]
    enable_pruning = context.run_config["client-enable-pruning"]
    pruning_rate = context.run_config["pruning-rate"]
    enable_quantization = context.run_config["client-enable-quantization"]
    bits_quantization = context.run_config["bits-quantization"]
    full_compression = context.run_config["full-compression"]
    model_name = context.run_config["model"]
    layer_compression = context.run_config["layer-compression"]
    init_space_quantization = context.run_config["init-space-quantization"]
    
    if model_name == "Net":
        net = Net()
    elif model_name == "ResNet12":
        net = ResNet12()
    elif model_name == "ResNet18":
        net = ResNet18()
    elif model_name == "QResNet12":
        net = QResNet12(bits_quantization)
    elif model_name == "QResNet18":
        net = QResNet18(bits_quantization)
    else:
        log(WARNING, "No existing model provided")
    
    client = FlowerClient(net,
                          model_name,
                          trainloader,
                          valloader,
                          learning_rate,
                          local_epochs,
                          enable_pruning,
                          pruning_rate,
                          enable_quantization,
                          bits_quantization,
                          partition_id,
                          layer_compression,
                          init_space_quantization,
                          full_compression)

    
    return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
