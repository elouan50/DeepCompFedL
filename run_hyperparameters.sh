#!/bin/bash

# We want to make a variation on:
# - pruning or not
# - quantization or not
# - Huffman encoding or not
# - number of clients epochs
# - batch size
# - learning rate


for number in 1 2 3
do
    for pr in 0 0.3
    do
        for qbits in 32 8
        do
            for epochs in 1 10
            do
                for bs in 32 8
                do
                    for lr in 0.1 0.01
                    do
                        echo "Experiment with pruning rate $pr, quantization on $qbits bits and $epochs local epochs."
                        flwr run --run-config "
                                        server-rounds=100
                                        client-enable-pruning=true
                                        pruning-rate=$pr
                                        client-enable-quantization=true
                                        bits-quantization=$qbits
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
        done
    done
done
