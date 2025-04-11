"""DeepCompFedL: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from deepcompfedl.strategy.DeepCompFedLStrategy import DeepCompFedLStrategy
from deepcompfedl.task import (
    get_weights,
    evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn,
)
from deepcompfedl.models.net import Net
from deepcompfedl.models.resnet12 import ResNet12
from deepcompfedl.models.resnet18 import ResNet18
from deepcompfedl.models.qresnet12 import QResNet12
from deepcompfedl.models.qresnet18 import QResNet18

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["server-rounds"]
    dataset = context.run_config["dataset"]
    client_epochs = context.run_config["client-epochs"]
    fraction_fit = context.run_config["fraction-fit"]
    aggregation_strategy = context.run_config["aggregation-strategy"]
    model_name = context.run_config["model"]
    enable_pruning = context.run_config["server-enable-pruning"]
    enable_quantization = context.run_config["server-enable-quantization"]
    pruning_rate = context.run_config["pruning-rate"]
    bits_quantization = context.run_config["bits-quantization"]
    layer_compression = context.run_config["layer-compression"]
    init_space_quantization = context.run_config["init-space-quantization"]
    full_compression = context.run_config["full-compression"]
    number = context.run_config["number"]
    save_online = context.run_config["save-online"]
    save_local = context.run_config["save-local"]
    alpha = context.run_config["alpha"]
    batch_size = context.run_config["batch-size"]
    learning_rate = context.run_config["learning-rate"]

    input_shape = {"MNIST": (1, 28, 28), "EMNIST": (1, 28, 28), "FEMNIST": (1, 28, 28), "CIFAR-10": (3, 32, 32), "ImageNet": (3, 64, 64)}
    num_classes = {"MNIST": 10, "EMNIST": 10, "FEMNIST": 62, "CIFAR-10": 10, "ImageNet": 200}
    
    # Initialize model parameters
    if model_name == "Net":
        model = Net()
    elif model_name == "ResNet12":
        model = ResNet12(input_shape=input_shape[dataset], num_classes=num_classes[dataset])
    elif model_name == "ResNet18":
        model = ResNet18(num_classes=num_classes[dataset])
    elif model_name == "QResNet12":
        model = QResNet12(bits_quantization)
    elif model_name == "QResNet18":
        model = QResNet18(bits_quantization)
    else:
        model = None
        print("Model not recognized")

    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    if aggregation_strategy == "DeepCompFedLStrategy":
        strategy = DeepCompFedLStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            num_rounds=num_rounds,
            dataset=dataset,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model=model_name,
            epochs=client_epochs,
            enable_pruning=enable_pruning,
            pruning_rate=pruning_rate,
            enable_quantization=enable_quantization,
            bits_quantization=bits_quantization,
            layer_compression=layer_compression,
            init_space_quantization=init_space_quantization,
            full_compression=full_compression,
            number=number,
            save_online=save_online,
            save_local=save_local,
        )
    elif aggregation_strategy == "FedAvg":
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
    else:
        print("Strategy not recognized")

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
