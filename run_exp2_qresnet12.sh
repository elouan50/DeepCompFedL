#!/bin/bash

# We want to make a variation on the quantization and the number of clients epochs

for number in 4 5 6
do
    for qbits in 8 4
    do
        for epochs in 1 10
        do
            echo "Experiment with pruning rate $prate and $epochs local epochs."
            flwr run --run-config "
                                server-rounds=100
                                client-enable-quantization=true
                                bits-quantization=$qbits
                                model='QResNet12'
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
