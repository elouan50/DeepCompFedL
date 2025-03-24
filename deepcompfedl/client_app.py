"""DeepCompFedL: A Flower / PyTorch app."""

import time
import torch
from logging import WARNING
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from flwr.common.logger import log

from deepcompfedl.compression.pruning import prune
from deepcompfedl.compression.quantization import quantize
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
                 layer_quantization,
                 init_space_quantization
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
        self.layer_quantization = layer_quantization
        self.init_space_quantization = init_space_quantization
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
        
        params = get_weights(self.net)
        
        after_training = time.perf_counter()
        train_time = after_training - begin
        
        ### Apply Pruning
        if self.enable_pruning:
            params = prune(params, self.pruning_rate)
        
        ### Apply Quantization
        if self.enable_quantization and self.model_name[0] != "Q":
            params = quantize(params,
                            self.bits_quantization,
                            self.layer_quantization,
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
    model_name = context.run_config["model"]
    layer_quantization = context.run_config["layer-quantization"]
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
                          layer_quantization,
                          init_space_quantization)

    
    return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
