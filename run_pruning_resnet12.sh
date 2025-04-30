#!/bin/bash

# For the pruning alone experiments

# We run it with the ResNet-12 on IID data

for number in 1 2 3
do
    for prate in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
    do
        for epochs in 1 10
        do
            echo "Experiment with pruning rate $prate and $epochs local epochs."
            flwr run --run-config "
                                server-rounds=100
                                client-enable-pruning=true
                                pruning-rate=$prate
                                model='ResNet12'
                                fraction-fit=0.4
                                aggregation-strategy='DeepCompFedLStrategy'
                                client-epochs=$epochs
                                number=$number
                                alpha=100
                                save-online=true
                                save-local=true
                                "
        done
    done
done
