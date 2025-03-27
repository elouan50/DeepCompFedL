"""DeepCompFedL: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, alpha: int | float, dataset: str, batch_size: int = 32):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                           partition_by="label",
                                           alpha=alpha,
                                           min_partition_size=1,
                                           self_balancing=True)
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
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, learning_rate, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
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
    
    # We weight all metrics by the number of examples
    
    # Training loss
    weighted_train_losses = [num_examples * metrics["train-loss"] for num_examples, metrics in fit_metrics]
    train_losses_aggregated = sum(weighted_train_losses) / num_total_fit_examples
    
    # Training time
    weighted_training_times = [num_examples * metrics["training-time"] for num_examples, metrics in fit_metrics]
    training_times_aggregated = sum(weighted_training_times) / num_total_fit_examples
    
    # Compression time
    weighted_compression_times = [num_examples * metrics["compression-time"] for num_examples, metrics in fit_metrics]
    compression_times_aggregated = sum(weighted_compression_times) / num_total_fit_examples
    
    return {"training-time": float(training_times_aggregated), "train-loss": float(train_losses_aggregated), "compression-time": float(compression_times_aggregated)}
