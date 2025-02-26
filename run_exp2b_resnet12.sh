#!/bin/bash

# We want to make a variation on the quantization bits and the number of clients epochs

for init_space in "random" "density" "uniform"
do
    for lq in "true" "false"
    do
        echo "Experiment with quantization on 4 bits and 1 local epoch."
        flwr run --run-config "
            server-rounds=30
            server-enable-quantization=true
            server-bits-quantization=4
            client-enable-quantization=true
            client-bits-quantization=4
            model='ResNet12'
            fraction-fit=0.4
            aggregation-strategy='DeepCompFedLStrategy'
            client-epochs=1
            init-space-quantization='$init_space'
            layer-quantization=$lq
            save-online=true
            "
    done
done
