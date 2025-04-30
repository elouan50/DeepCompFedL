#!/bin/bash

# We want to make a variation on the pruning rate and the number of clients epochs

for number in 1 2 3
do
    for prate in 0.25 0.5 0.6 0.7 0.8 0.9
    do
        for qbits in 8 6 5 4 3
        do
            for fc in true false
            do
                echo "Experiment with pruning rate $prate and $qbits quantization."
                flwr run --run-config "
                    server-rounds=50
                    client-enable-pruning=true
                    pruning-rate=$prate
                    client-enable-quantization=true
                    bits-quantization=$qbits
                    full-compression=$fc
                    model='Net'
                    dataset='FEMNIST'
                    fraction-fit=0.4
                    aggregation-strategy='DeepCompFedLStrategy'
                    client-epochs=1
                    number=$number
                    alpha=100
                    save-online=true
                    save-local=true
                    "
            done
        done
    done
done
