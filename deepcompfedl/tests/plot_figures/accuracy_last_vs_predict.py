"""
This file aims to calculate all the metrics
relative to the third experiment.
It includes evaluating average values, show final performances,
and interpolating convergence speed.
"""

import numpy as np
import pandas as pd
import wandb
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-scenario3"
nrounds = 350

print("---------------------------")
print("------  Scenario 3  -------")
print("---------------------------")
print("")
print("Model: ResNet-12")
print("Dataset: CIFAR-10")
print("non-IID data")
print("Prune with 30%")
print("Quantization on 256 centroids")
print("")

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

baseline = np.zeros((nrounds))
accuracies = np.zeros((3, nrounds))

runs = [
    "elouan50-rwth-aachen-university/deepcompfedl-scenario3/runs/cfl3hyxb",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario3/runs/qf3kmfdw",
    "elouan50-rwth-aachen-university/deepcompfedl-scenario3/runs/9jgu89ia"
    ]

i = 0
for run_path in runs:
    
    run = api.run(run_path)
    df = run.history(keys=["accuracy"])

    accuracies[i] += np.array(df['accuracy'])/len(runs)
    i += 1
    
steps = np.array([i for i in range(1, nrounds+1)])


accuracy = (accuracies[0, :] + accuracies[1, :] + accuracies[2, :])
min_, max_ = min(accuracy), max(accuracy)
K = max_ - min_
tau = steps[np.argmax(accuracy > 0.63*K+min_)]
bias = min_
accuracy_calc = first_order_response(steps, K, tau, bias)
r2 = 1-np.sum((accuracy - accuracy_calc)**2)/np.sum((accuracy - np.mean(accuracy))**2)


params, _ = curve_fit(first_order_response, steps, accuracy, p0=[max_-min_, steps[np.argmax(accuracy > 0.63*(max_-min_))], min_])
K_opt, tau_opt, bias_opt = params
accuracy_fit = first_order_response(steps, K_opt, 1 if int(tau_opt)==0 else int (tau_opt), bias_opt)
r2_fit = 1-np.sum((accuracy - accuracy_fit)**2)/np.sum((accuracy - np.mean(accuracy))**2)

# Plot the accuracy values
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracy, label='average real accuracy', color='r')
plt.plot(steps, accuracy_calc, label='Hand calculated model', color='b')
plt.plot(steps, accuracy_fit, label='scipy calculated model', color='g')

# Labels and title
plt.xlabel('Step')
plt.xlim(0, 350)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Training Accuracy Across Steps')
plt.legend()
plt.grid()
plt.figtext(0.4, 0.35, f"Baseline final accuracy: {100*accuracy[-1]:.3f}%", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
plt.figtext(0.4, 0.25, f"hand: τ = {int(tau)}, C = {K:.3f}, bias = {bias:.3f}, final value {100*(K+bias):.3f}%", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
plt.figtext(0.4, 0.15, f"scipy: τ = {int(tau_opt)}, C = {K:.3f}, bias = {bias_opt:.3f}, final value {100*(K_opt+bias_opt):.3f}%", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})

# Save the plot
plt.savefig("/home/colybes/Downloads/plot.png")
