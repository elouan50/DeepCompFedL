"""DeepCompFedL: A Flower / PyTorch app."""

import os
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from deepcompfedl.compression.pruning import prune
from deepcompfedl.compression.quantization import quantize_layers, quantize_model
from deepcompfedl.compression.encoding import encode
from deepcompfedl.compression.decoding import decode
from deepcompfedl.compression.metrics import pruned_weights, quantized_model, quantized_layers

from deepcompfedl.task import (
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)
from deepcompfedl.models.net import Net
from deepcompfedl.models.resnet12 import ResNet12

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, enable_pruning, pruning_rate, enable_quantization, bits_quantization, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.enable_pruning = enable_pruning
        self.pruning_rate = pruning_rate
        self.enable_quantization = enable_quantization
        self.bits_quantization = bits_quantization
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        
        ### Apply Pruning
        if self.enable_pruning:
            params = get_weights(self.net)
            parameters = prune(params, self.pruning_rate)
            set_weights(self.net, parameters)
            if self.partition_id == 0:
                print(f"Effective sent pruning (for client {self.partition_id} in fit):")
                pruned_weights(self.net)
                print("")
        
        ### Apply Quantization
        if self.enable_quantization:
            quantize_layers(self.net, self.bits_quantization)
            if self.partition_id == 0:
                print(f"Effective received quantization (for client {self.partition_id}):")
                quantized_layers(self.net)
                print("")
        
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        ### Apply Pruning
        if self.enable_pruning:
            ## Print stats for pruning
            if self.partition_id == 0:
                print(f"Effective received pruning (for client {self.partition_id} in client_fn):")
                pruned_weights(self.net)
                print("")
                
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    ### Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    dataset = context.run_config["dataset"]
    trainloader, valloader = load_data(partition_id, num_partitions, dataset)
    local_epochs = context.run_config["client-epochs"]
    enable_pruning = context.run_config["client-enable-pruning"]
    pruning_rate = context.run_config["client-pruning-rate"]
    enable_quantization = context.run_config["client-enable-quantization"]
    bits_quantization = context.run_config["client-bits-quantization"]
    model_name = context.run_config["model"]
    
    if model_name == "Net":
        net = Net()
    elif model_name == "ResNet12":
        net = ResNet12(16, (3,32,32), 10)
    else:
        net = None

    # ### Create saving directory
    # save_dir = "deepcompfedl/saves/"
    # os.makedirs(save_dir, exist_ok=True)
    
    client = FlowerClient(net,
                          trainloader,
                          valloader,
                          local_epochs,
                          enable_pruning,
                          pruning_rate,
                          enable_quantization,
                          bits_quantization,
                          partition_id)

    ### Apply Pruning
    # if enable_pruning:
    #     ## Print stats for pruning
    #     if partition_id == 0:
    #         print(f"Effective received pruning (for client {partition_id} in client_fn):")
    #         pruned_weights(client.net)
    #         print("")
    
    
    ### Apply Quantization
    ## Layer-wise
    if enable_quantization:
        if partition_id == 0:
            print(f"Effective received quantization (for client {partition_id}):")
            quantized_layers(client.net)
            print("")
    
    ## Model-wise
    # if partition_id == 0:
    #     print(f"Effective quantization (for client {partition_id}):")
    #     quantized_model(client.net)
    #     print("")

    
    
    ### Return Encoded Client
    # encode(client)
    
    return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
