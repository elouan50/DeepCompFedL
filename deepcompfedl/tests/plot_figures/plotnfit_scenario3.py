"""
This file aims to calculate all the metrics
relative to the third experiment.
It includes evaluating average values, show final performances,
and interpolating convergence speed.
"""

import numpy as np
import pandas as pd
import wandb

api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-scenario3"
nrounds = 100

print("---------------------------")
print("------  Scenario 3  -------")
print("---------------------------")
print("")
print("Model: ResNet-12")
print("Dataset: CIFAR-10")
print("non-IID data")
print("")

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

baseline = np.zeros((nrounds))
accuracies = np.zeros((7, 10, nrounds))

pr_dic = {0.1: 0,
          0.2: 1,
          0.3: 2,
          0.4: 3,
          0.5: 4,
          0.6: 5,
          0.7: 6,
          0.8: 7,
          0.9: 8,
          0.95: 9
          }

runs = api.runs(project)

for run in runs:
    
    run_name = run.name
    pr = run.config["pruning-rate"]
    qb = run.config["bits-quantization"]
    df = run.history(keys=["accuracy"])

    if qb==32: #Baseline
        baseline += np.array(df['accuracy'])
    else:
        accuracies[qb-2, pr_dic[pr]] += np.array(df['accuracy'])
    
steps = np.array([i for i in range(1, nrounds+1)])

min_, max_ = min(baseline), max(baseline)

# Own method to calculate the model
K = max_ - min_
tau = steps[np.argmax(baseline > 0.63*K+min_)]
bias = min_

# Calculate the model and evaluate it
accuracy_calc = first_order_response(steps, K, tau, bias)
r2 = 1-np.sum((baseline - accuracy_calc)**2)/np.sum((baseline - np.mean(baseline))**2)



print("")
print("Baseline ACCURACY: ", round(baseline[nrounds-1]/3, 5))
print("")
print("           |                                  Pruning rate")
print("Nb clusters|   0.1     0.2     0.3     0.4     0.5     0.6     0.7     0.8     0.9     0.95")
print("-----------+----------------------------------------------------------------------------------")

for qb in range(6, -1, -1):
    print("   ", 2**(qb+2), "" if qb>=5 else (" " if qb >= 2 else "  "), "  | ", end="")
    for pr in range(10):
        print(str(round(accuracies[qb, pr, nrounds-1]/3, 5)).ljust(7, "0"), end=" ")
    print("")

print("")
print("Baseline TAU: ", tau)
print("Baseline r²: ", r2)
print("")
print("           |                                  Pruning rate")
print("Nb clusters| 0.1      0.2      0.3      0.4      0.5      0.6      0.7      0.8      0.9      0.95")
print("-----------+------------------------------------------------------------------------------------------")

for qb in range(6, -1, -1):
    print("   ", 2**(qb+2), "" if qb>=5 else (" " if qb >= 2 else "  "), "  | ", end="")
    for pr in range(10):
        accuracy = accuracies[qb, pr] /3
        min_, max_ = min(accuracy), max(accuracy)
        K = max_ - min_
        tau = steps[np.argmax(accuracy > 0.63*K+min_)]
        bias = min_
        accuracy_calc = first_order_response(steps, K, tau, bias)
        r2 = 1-np.sum((accuracy - accuracy_calc)**2)/np.sum((accuracy - np.mean(accuracy))**2)

        print(str(tau).rjust(2, " ") + " (" + str(round(r2, 2))[1:].ljust(3, "0")+ ")", end=" ")
    print("")

print("")

