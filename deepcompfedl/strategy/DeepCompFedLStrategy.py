"""DeepCompFedL: A Flower / PyTorch app."""

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import wandb
import torch
import os

from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

from deepcompfedl.compression.pruning import prune
from deepcompfedl.compression.quantization import quantize
from deepcompfedl.task import set_weights
from deepcompfedl.models.resnet18 import ResNet18
from deepcompfedl.models.resnet12 import ResNet12

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class DeepCompFedLStrategy(FedAvg):
    """Costumized Federated Averaging strategy.
    
    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    num_rounds : int (default: 3)
        Number of server rounds.
    dataset : str (default: "")
        Dataset used for training and evaluation of the model.
    model : str (default: "")
        Model that will be trained and evaluated.
    epochs : int (default: 1)
        Number of local epochs per client at each round.
    enable_pruning : bool (default: False)
        Enable (True) or disable (False) pruning of model updates, optional. Defaults to False.
    pruning_rate : float (default: 0.)
        Pruning rate (only if pruning is enabled), optional.
    enable_quantization : bool (default: False)
        Enable (True) or disable (False) quantization of model updates, optional. Defaults to False.
    bits_quantization : int (default: 32)
        Number of bits to represent the quantized model, optional.
    layer_quantization : bool (default: True)
        Enable (True) or disable (False) layer-wise quantization, optional. Defaults to True.
    init_space_quantization : str (default: "uniform")
        Initialization space for the K-Means clustering, optional. Defaults to "uniform".
    number : int (default: 1)
        ID of the pass we call the experiment
    save_online : bool (default: False)
        Save the results online, optional. Defaults to False.
    save_local : bool (default: False)
        Save the results locally, optional. Defaults to False.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        num_rounds: int = 3,
        dataset: str = "",
        alpha: int | bool = 100,
        model: str = "",
        epochs: int = 1,
        enable_pruning: bool = False,
        pruning_rate: float = 0.,
        enable_quantization: bool = False,
        bits_quantization: int = 32,
        layer_quantization: bool = True,
        init_space_quantization: str = "uniform",
        number: int = 1,
        save_online: bool = False,
        save_local: bool = False,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.model = model
        self.epochs = epochs
        self.num_rounds = num_rounds
        self.number = number
        self.save_online = save_online
        self.save_local = save_local
        self.enable_pruning = enable_pruning
        self.pruning_rate = pruning_rate
        self.enable_quantization = enable_quantization
        self.bits_quantization = bits_quantization
        self.layer_quantization = layer_quantization
        self.init_space_quantization = init_space_quantization
        
        if save_online:
            wandb.init(
                project="deepcompfedl-quantization",
                id=f"q{bits_quantization}-i{init_space_quantization}-l{layer_quantization}-e{epochs}-n{number}",
                config={
                    "aggregation-strategy": "DeepCompFedLStrategy",
                    "num-rounds": num_rounds,
                    "dataset": dataset,
                    "alpha": alpha,
                    "model": model,
                    "epochs": epochs,
                    "fraction-fit": fraction_fit,
                    "server-enable-pruning": enable_pruning,
                    "server-pruning-rate": pruning_rate,
                    "server-enable-quantization": enable_quantization,
                    "server-bits-quantization": bits_quantization,
                    "server-layer-quantization": layer_quantization,
                    "server-init-space-quantization": init_space_quantization,
                },
            )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
        
        # Prune the parameters sent to client
        if self.enable_pruning:
            aggregated_ndarrays = prune(aggregated_ndarrays, self.pruning_rate)
        
        if self.enable_quantization:
            aggregated_ndarrays = quantize(aggregated_ndarrays,
                                            self.bits_quantization,
                                            self.layer_quantization,
                                            self.init_space_quantization)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if server_round == self.num_rounds and self.save_local:
            save_dir = "deepcompfedl/saves/exp2quant"

            os.makedirs(save_dir, exist_ok=True)
            
            if self.model == "ResNet18":
                model = ResNet18()
                set_weights(model, aggregated_ndarrays)
                torch.save(model, f"{save_dir}/q{self.bits_quantization}-i{self.init_space_quantization}-l{self.layer_quantization}-e{self.epochs}-n{self.number}.ptmodel")
            elif self.model == "ResNet12":
                model = ResNet12()
                set_weights(model, aggregated_ndarrays)
                torch.save(model, f"{save_dir}/q{self.bits_quantization}-i{self.init_space_quantization}-l{self.layer_quantization}-e{self.epochs}-n{self.number}.ptmodel")
            else:
                log(WARNING, "Model couldn't be saved")

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            if self.save_online:
                wandb.log(metrics_aggregated, step=server_round)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            if self.save_online:
                wandb.log(metrics_aggregated, step=server_round)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
