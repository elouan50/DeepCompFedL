#!/bin/bash



for number in 1 2 3
do
    for qbits in 32 8 4
    do
        echo "Experiment with quantization on $qbits bits."
        flwr run --run-config "
                server-rounds=300
                client-enable-quantization=true
                bits-quantization=$qbits
                model='ResNet18'
                fraction-fit=0.1
                aggregation-strategy='DeepCompFedLStrategy'
                client-epochs=1
                number=$number
                alpha=100
                save-online=true
                save-local=true
                "
    done
done