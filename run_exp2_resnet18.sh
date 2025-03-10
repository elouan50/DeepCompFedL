#!/bin/bash

# We want to make a variation on the pruning rate.

for number in 1 2 3
do
    for qbits in 32 8 4
    do
        for batch in 8
        do
            echo "Experiment with quantization on $qbits bits and batch size $batch."
            flwr run --run-config "
                                server-rounds=1000
                                client-enable-quantization=true
                                bits-quantization=$qbits
                                model='ResNet18'
                                fraction-fit=0.1
                                aggregation-strategy='DeepCompFedLStrategy'
                                client-epochs=1
                                number=$number
                                alpha=100
                                batch-size=$batch
                                save-online=true
                                save-local=true
                                "
        done
    done
done