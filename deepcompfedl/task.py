"""DeepCompFedL: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


dict_tranforms = {  
    "cifar10"           : Compose([
                                        RandomCrop(32, padding=4),
                                        RandomHorizontalFlip(),
                                        ToTensor(),
                                        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    "emnist"            : Compose([
                                        ToTensor(),
                                        Normalize((0.1307,), (0.3081,))]), 
    "cifar100"          : Compose([
                                        RandomCrop(32, padding=4),
                                        RandomHorizontalFlip(),
                                        ToTensor(),
                                        Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]), }

dict_tranforms_test = {
    "cifar10"           : Compose([
                                        ToTensor(),
                                        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    "emnist"            : Compose([
                                        ToTensor(),
                                        Normalize((0.1307,), (0.3081,))]),
    "cifar100"          : Compose([
                                        ToTensor(),
                                        Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]), }


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, dataset: str):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def evaluate_metrics_aggregation_fn(eval_metrics):
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in eval_metrics)
    weighted_accuracies = [num_examples * metrics["accuracy"] for num_examples, metrics in eval_metrics]
    metrics_aggregated = sum(weighted_accuracies) / num_total_evaluation_examples
    return {"accuracy": float(metrics_aggregated)}

def fit_metrics_aggregation_fn(fit_metrics):
    num_total_fit_examples = sum(num_examples for (num_examples, _) in fit_metrics)
    weighted_accuracies = [num_examples * metrics["time"] for num_examples, metrics in fit_metrics]
    metrics_aggregated = sum(weighted_accuracies) / num_total_fit_examples
    return {"time": float(metrics_aggregated)}
