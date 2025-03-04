#!/bin/bash

# We want to make a variation on the pruning rate.

for number in 1 2 3
do
    for prate in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
    do
        echo "Experiment with pruning rate $prate"
        flwr run --run-config "
                            server-rounds=1000
                            client-enable-pruning=true
                            client-pruning-rate=$prate
                            model='ResNet18'
                            fraction-fit=0.1
                            aggregation-strategy='DeepCompFedLStrategy'
                            client-epochs=1
                            alpha=1
                            number=$number
                            save-online=true
			                save-local=true
"
    done
done
