"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import pandas as pd
import wandb

api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-quantization"

accuracies = np.zeros((3, 2, 100))
training_times = np.zeros((3, 2, 100))

qb_dic = {32: 0, 8: 1, 4: 2}
qb_dic_inv = {0: "32", 1: "8", 2: "4"}


runs = api.runs(project)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

for run in runs:
	if "iuniform-lTrue" in run.name:
		# Get the history of the run
		c = run.config
		if c["server-bits-quantization"] > 1:
			df = run.history(keys=[
								"accuracy",
								"training-time"
								])
			print(df.size, run.name)


			# Extract data
			accuracies[
					qb_dic[c["server-bits-quantization"]],
					int((c["epochs"]-1)/9)
					] += np.array(df['accuracy'])
			training_times[
					qb_dic[c["server-bits-quantization"]],
					int((c["epochs"]-1)/9)
					] += np.array(df['training-time'])

steps = np.array([i for i in range(1,101)])

for qb in range(3):
	for ep in range(2):
		accuracy = accuracies[qb, ep] /3
		t_train = training_times[qb, ep] /3

		min_, max_ = min(accuracy), max(accuracy)

		# Own method to calculate the model
		K = max_ - min_
		tau = steps[np.argmax(accuracy > 0.63*K+min_)]
		bias = min_

		# Calculate the model and evaluate it
		accuracy_calc = first_order_response(steps, K, tau, bias)
		r2 = 1-np.sum((accuracy - accuracy_calc)**2)/np.sum((accuracy - np.mean(accuracy))**2)

		print("---------------------------")
		print("Settings:")
		print("Quantization:   ", qb_dic_inv[qb])
		print("Epochs:         ", int(ep*9+1))
		print("")
		print("Results:")
		print("Final accuracy: ", accuracy[-1])
		print("Avg t_train:    ", np.mean(t_train))
		print("tau:            ", tau)
		print("r2:  ", r2)
		print("")
