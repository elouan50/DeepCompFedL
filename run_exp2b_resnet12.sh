#!/bin/bash

# We want to make a variation on the quantization bits and the number of clients epochs

for number in 1 2 3
do
    for qbits in 32 8 4 1
    do
        for epochs in 1 10
        do
            for init_space in "random" "density" "uniform"
            do
                for lq in "true" "false"
                do
                    echo "Experiment with quantization on $qbits bits and $epochs local epochs."
                    flwr run --run-config "
                                        server-rounds=100
                                        server-enable-quantization=true
                                        server-bits-quantization=$qbits
                                        client-enable-quantization=true
                                        client-bits-quantization=$qbits
                                        model='ResNet12'
                                        fraction-fit=0.4
                                        aggregation-strategy='DeepCompFedLStrategy'
                                        client-epochs=$epochs
                                        number=$number
                                        init-space-quantization='$init_space'
                                        layer-quantization=$lq
                                        save-online=true
                                        save-local=true
                                        "
                done
            done
        done
    done
done
