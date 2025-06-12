#!/bin/bash

# For the FCP experiments:
# Run it in the third described scenario

for number in 1 2 3
do
    for prate in 0.0 0.2 0.4 0.5 0.6 0.9
    do
        for qbits in 32 8 4
        do
            for fc in true false
            do
                echo "Experiment with pruning rate $prate and $qbits quantization."
                flwr run --run-config "
                    server-rounds=100
                    client-enable-pruning=true
                    pruning-rate=$prate
                    client-enable-quantization=true
                    bits-quantization=$qbits
                    full-compression=$fc
                    model='ResNet12'
                    dataset='CIFAR-10'
                    fraction-fit=0.2
                    aggregation-strategy='DeepCompFedLStrategy'
                    client-epochs=1
                    number=$number
                    alpha=1
                    save-online=true
                    save-local=true
                    "
            done
        done
    done
done
