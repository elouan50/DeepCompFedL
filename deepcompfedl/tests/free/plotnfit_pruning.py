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
project = "elouan50-rwth-aachen-university/deepcompfedl-pruning-layer-or-model"

accuracies = np.zeros((2,12,100))
rnd_time = np.zeros((2,12,100))

runs = api.runs(project)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

for run in runs:
   if int(run.name[-1])>3:
      df = run.history(keys=["accuracy", "round-time"])
      c = run.config

      pr = float(c["pruning-rate"])
      if pr<0.95:
         pr = int(pr*10)
      elif pr==0.95:
         pr = 10
      elif pr==0.99:
         pr = 11
      else:
         break

      # Extract data
      accuracies[int(c["layer-compression"]), pr] += np.array(df['accuracy'])
      rnd_time[int(c["layer-compression"]), pr] += np.array(df['round-time'])
      
steps = np.array([i for i in range(1,101)])

for lc in range(2):
   for pr in range(12):
      accuracy = accuracies[lc, pr] / 3
      rt = rnd_time[lc, pr] / 3

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
      print("Pruning:       ", pr*0.1 if pr<10 else round(pr*0.04+0.55,2))
      print("Layer-compress:", bool(lc))
      print("")
      print("Results:")
      print("Final accuracy: ", accuracy[-1])
      print("Avg round time: ", np.mean(rt[1:]))
      print("tau:            ", tau)
      print("r2:  ", r2)
      print("")
               
