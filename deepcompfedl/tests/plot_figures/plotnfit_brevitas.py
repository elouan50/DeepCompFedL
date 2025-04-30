"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import pandas as pd
import wandb

api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-brevitas"

accuracies = np.zeros((2, 2, 100))
training_times = np.zeros((2, 2, 100))
compression_times = np.zeros((2, 2, 100))
round_times = np.zeros((2, 2, 100))
communication_overheads = np.zeros((2, 2, 100))

runs = api.runs(project)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

for run in runs:
	if run.name[-1] == "1":
		# Get the history of the run
		c = run.config
		df = run.history(keys=[
							"accuracy",
							"training-time",
							"compression-time",
							"round-time",
							"communication-overhead"
							])
		print(df.size, run.name)


		# Extract data
		accuracies[
				int((c["bits-quantization"]-1)/4),
				int((c["epochs"]-1)/9)
				] += np.array(df['accuracy'])
		training_times[
				int((c["bits-quantization"]-1)/4),
				int((c["epochs"]-1)/9)
				] += np.array(df['training-time'])
		compression_times[
				int((c["bits-quantization"]-1)/4),
				int((c["epochs"]-1)/9)
				] += np.array(df['compression-time'])
		round_times[
				int((c["bits-quantization"]-1)/4),
				int((c["epochs"]-1)/9)
				] += np.array(df['round-time'])
		communication_overheads[
				int((c["bits-quantization"]-1)/4),
				int((c["epochs"]-1)/9)
				] += np.array(df['communication-overhead'])

steps = np.array([i for i in range(1,101)])

for qb in range(2):
	for ep in range(2):
		accuracy = accuracies[qb, ep]
		t_train = training_times[qb, ep]
		t_compress = compression_times[qb, ep]
		t_round = round_times[qb, ep]
		co = communication_overheads[qb, ep]

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
		print("Quantization:   ", int(qb*4+4))
		print("Epochs:         ", int(ep*9+1))
		print("")
		print("Results:")
		print("Final accuracy: ", accuracy[-1])
		print("Avg t_train:    ", np.mean(t_train))
		print("Avg t_compress: ", np.mean(t_compress))
		print("Avg t_round:    ", np.mean(t_round))
		print("Avg t_compute:  ", np.mean(t_round*co))
		print("tau:            ", tau)
		print("r2:  ", r2)
		print("")
