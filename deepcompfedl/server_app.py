"""DeepCompFedL: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from deepcompfedl.task import Net, ResNet12, get_weights


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    aggregation_strategy = context.run_config["aggregation-strategy"]
    model_name = context.run_config["model"]

    # Initialize model parameters
    if model_name == "Net":
        model = Net()
    elif model_name == "ResNet12":
        model = None
    else:
        model = None
        print("Model not recognised")

    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    if aggregation_strategy == "FedAvg":
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
        )
    else:
        print("Strategy not recognised")

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
