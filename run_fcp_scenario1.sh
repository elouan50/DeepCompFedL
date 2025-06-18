#!/bin/bash

# For the FCP experiments:
# Run it in the first described scenario
# Dataset: CIFAR-10 ; IID

# Baseline
for number in 1 2 3
do
    echo "Baseline experiment."
    flwr run --run-config "
        server-rounds=100
        aggregation-strategy='DeepCompFedLStrategy'
        model='ResNet12'
        dataset='CIFAR-10'
        fraction-fit=0.4
        client-epochs=1
        alpha=100
        number=$number
        project-name='deepcompfedl-scenario1'
        save-online=true
        save-local=true
    "
done

# Main experiments
for number in 1 2 3
do
    for qbits in 8 4
    do
        for prate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
        do
            echo "NEW EXPERIMENT"
            echo "Pruning rate: $prate"
            echo "Quantization bits: $qbits"
            flwr run --run-config "
                server-rounds=100
                aggregation-strategy='DeepCompFedLStrategy'
                model='ResNet12'
                dataset='CIFAR-10'
                fraction-fit=0.4
                client-epochs=1
                full-compression=true
                pruning-rate=$prate
                bits-quantization=$qbits
                alpha=100
                number=$number
                project-name='deepcompfedl-scenario1'
                save-online=true
                save-local=true
            "
        done
    done
done

# Processing the results
python3 ./deepcompfedl/tests/plot_figures/plotnfit_scenario1.py
