import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, Optional, Tuple, List
import numpy as np
from collections import OrderedDict
from prettytable import PrettyTable
import brevitas.nn as qnn


def pile_str(line,item):
    return "_".join([line,item])

def get_tensor_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])


def tell_history(hist,file_name : str, infos = None,path : str = "",head=None):

    _, accuracy = zip(*hist.metrics_centralized["accuracy"])
    losses_cent = hist.losses_centralized
    losses_dis = hist.losses_distributed
    accuracy = np.asarray(accuracy)

    infos["accuracy"] = accuracy
    infos["losses_cent"] = losses_cent
    infos["losses_dis"] = losses_dis

    with open(path+file_name+".npy","wb") as f:
        np.save(f,infos)

# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str,
          lr : float = 0.01, momentum : float = 0.9,
          bnn : bool = False):

    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if bnn:
                net.binarization()
            loss = criterion(net(images), labels)
            loss.backward()
            
            if bnn:
                net.restore()

            optimizer.step()

            if bnn:
                net.clip()            

# borrowed from Pytorch quickstart example
def test(net, testloader, device: str, bnn : bool = False):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        if bnn:
            net.binarization()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

def total_sum_params(iter):
    return sum(
        param.numel() for param in iter.parameters()
    )

def get_model_size(model,wbits):

    total_params = total_sum_params(model)
    quant_params = 0

    for layer in model.modules() :
        if(isinstance(layer,qnn.QuantConv2d) or isinstance(layer,qnn.QuantLinear)):
            quant_params += total_sum_params(layer)

    fp_params = total_params-quant_params

    quant_size = quant_params*wbits
    model_size = quant_size + fp_params*32
    
    # no format (kiB or MiB)
    return model_size,total_params,quant_size,quant_params 

def pretty_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
