# !/bin/bash

# For the pruning experiments
#
# We want to evaluate the best setting:
# between layer- and model-wise pruning

for number in 1 2 3
do
    for prate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
    do
        for lcomp in true false
        do
            echo "Experiment with pruning on layers or whole model."
            flwr run --run-config "
                server-rounds=100
                aggregation-strategy='DeepCompFedLStrategy'
                model='ResNet12'
                fraction-fit=0.4
                client-epochs=1
                layer-compression=$lcomp
                client-enable-pruning=true
                pruning-rate=$prate
                alpha=100
                number=$number
                save-online=true
                save-local=true
            "
        done
    done
done