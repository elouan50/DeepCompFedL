"""DeepCompFedL: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from deepcompfedl.compression.pruning import prune
from deepcompfedl.compression.quantization import quantize_layers
from deepcompfedl.compression.encoding import encode
from deepcompfedl.compression.decoding import decode
from deepcompfedl.compression.metrics import pruned_weights

from deepcompfedl.compression.utils import pretty_parameters

from deepcompfedl.task import (
    Net,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
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
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["client-epochs"]
    pruning_rate = context.run_config["client-pruning-rate"]
    bits_quantization = context.run_config["client-bits-quantization"]

    client = FlowerClient(net, trainloader, valloader, local_epochs)

    # Apply Pruning
    prune(client.net, pruning_rate)
    
    # Print stats for pruning
    # print(f"Effective pruning (for client {partition_id}):")
    # pruned_weights(client.net)
    # print("")
    
    # Apply Quantization
    quantize_layers(client.net, bits_quantization)

    if partition_id==0:
        pretty_parameters(client.net)
    
    
    # Return Encoded Client
    # encode(client)
    
    # pretty_parameters(client.net)
    
    return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
