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


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["server-rounds"]
    dataset = context.run_config["dataset"]
    client_epochs = context.run_config["client-epochs"]
    fraction_fit = context.run_config["fraction-fit"]
    aggregation_strategy = context.run_config["aggregation-strategy"]
    model_name = context.run_config["model"]
    enable_pruning = context.run_config["server-enable-pruning"]
    pruning_rate = context.run_config["server-pruning-rate"]
    enable_quantization = context.run_config["server-enable-quantization"]
    bits_quantization = context.run_config["server-bits-quantization"]
    number = context.run_config["number"]

    # Initialize model parameters
    if model_name == "Net":
        model = Net()
    elif model_name == "ResNet12":
        model = ResNet12(16, (3,32,32), 10)
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
            model=model_name,
            epochs=client_epochs,
            enable_pruning=enable_pruning,
            pruning_rate=pruning_rate,
            enable_quantization=enable_quantization,
            bits_quantization=bits_quantization,
            number=number,
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
