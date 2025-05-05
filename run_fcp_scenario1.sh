#!/bin/bash

# For the FCP experiments:
# Run it in the first described scenario

for number in 1 2 3
do
    for qbits in 32 8 4 3
    do
        for prate in 0.0 0.2 0.5 0.95
        do
            echo "Experiment with quantization on $qbits bits and 1 local epochs."
            flwr run --run-config "
                server-rounds=100
                aggregation-strategy='DeepCompFedLStrategy'
                model='ResNet12'
                fraction-fit=0.4
                client-epochs=1
                full-compression=true
                pruning-rate=$prate
                bits-quantization=$qbits
                alpha=100
                number=$number
                save-online=true
                save-local=true
            "
        done
    done
done
