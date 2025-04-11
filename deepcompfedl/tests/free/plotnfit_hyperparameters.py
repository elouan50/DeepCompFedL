"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import wandb

api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-hyperparameters"

accuracies = np.zeros((2,2,2,2,2,100))
rnd_time = np.zeros((2,2,2,2,2,100))

runs = api.runs(project)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

for run in runs:
   df = run.history(keys=["accuracy", "round-time"])
   c = run.config

   # Extract data
   accuracies[int(c["pruning-rate"]==0.3),
            int(c["bits-quantization"]==8),
            int(c["epochs"]==10),
            int(c["learning-rate"]==0.01),
            int(c["batch-size"]==32)] += np.array(df['accuracy'])
   
   
   rnd_time[int(c["pruning-rate"]==0.3),
            int(c["bits-quantization"]==8),
            int(c["epochs"]==10),
            int(c["learning-rate"]==0.01),
            int(c["batch-size"]==32)] += np.array(df['round-time'])
   
steps = np.array([i for i in range(1,101)])

for pr in range(2):
   for qb in range(2):
      for ep in range(2):
         for lr in range(2):
            for bs in range(2):
               accuracy = accuracies[pr, qb, ep, lr, bs] / 3
               rt = rnd_time[pr, qb, ep, lr, bs] / 3

               min_, max_ = min(accuracy), max(accuracy)

               # Own method to calculate the model
               K = max_ - min_
               tau = steps[np.argmax(accuracy > 0.63*K+min_)] 
               bias = min_

               # Calculate the model and evaluate it
               accuracy_calc = first_order_response(steps, K, tau, bias)
               mse = np.mean((accuracy - accuracy_calc) ** 2)
               r2 = 1-np.sum((accuracy - accuracy_calc)**2)/np.sum((accuracy - np.mean(accuracy))**2)

               print("---------------------------")
               print("Settings:")
               print("Pruning:       ", bool(pr))
               print("Quantization:  ", bool(qb))
               print("Epochs:        ", 10 if ep else 1)
               print("Learning rate: ", 0.01 if lr else 0.1)
               print("Batch size:    ", 32 if bs else 8)
               print("")
               print("Results:")
               print("Final accuracy: ", accuracy[-1])
               print("Avg round time: ", np.mean(rt[1:]))
               print("tau:            ", tau)
               print("r2:  ", r2)
               print("")
               
