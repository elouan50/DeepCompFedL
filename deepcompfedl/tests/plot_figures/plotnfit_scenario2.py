"""
This file aims to calculate all the metrics
relative to the first experiment.
It includes evaluating average values, show final performances,
and interpolating convergence speed.
"""

import numpy as np
import pandas as pd
import wandb

api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-scenario2"

print("---------------------------")
print("------  Scenario 2  -------")
print("---------------------------")
print("")
print("Model: ResNet-12")
print("Dataset: FEMNIST")
print("IID data")
print("")

# Baseline init
size = np.zeros((1,100))
accuracy = np.zeros((1,100))
train_loss = np.zeros((1,100))
t_select = np.zeros((1,100))
t_train = np.zeros((1,100))
t_compress = np.zeros((1,100))
t_decode = np.zeros((1,100))
t_aggregate = np.zeros((1,100))
t_round = np.zeros((1,100))

# Experiments init
sizes = np.zeros((10,2,100))
train_losses = np.zeros((10,2,100))
accuracies = np.zeros((10,2,100))
at_select = np.zeros((10,2,100))
at_train = np.zeros((10,2,100))
at_compress = np.zeros((10,2,100))
at_decode = np.zeros((10,2,100))
at_aggregate = np.zeros((10,2,100))
at_round = np.zeros((10,2,100))

runs = api.runs(project)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

for run in runs:
	# Get the history of the run
	c = run.config
	df = run.history(keys=[
						"model-size",
						"accuracy",
						"train-loss",
						"t_select",
						"t_train",
						"t_compress",
						"t_decode",
						"t_aggregate",
						"t_round"
						])

	if not c["full-compression"]:
		# Baseline
		size += np.array(df["model-size"])
		accuracy += np.array(df["accuracy"])
		train_loss += np.array(df["train-loss"])
		t_select += np.array(df["t_select"])
		t_train += np.array(df["t_train"])
		t_compress += np.array(df["t_compress"])
		t_decode += np.array(df["t_decode"])
		t_aggregate += np.array(df["t_aggregate"])
		t_round += np.array(df["t_round"])

	else:
		# Experiments
		pr = c["pruning-rate"]
		if pr > 0.9:
			pr = 0
		else:
			pr = int(pr*10)

		qb = int(c["bits-quantization"]/4 - 1)

		sizes[pr, qb] += np.array(df["model-size"])
		accuracies[pr, qb] += np.array(df["accuracy"])
		train_losses[pr, qb] += np.array(df["train-loss"])
		at_select[pr, qb] += np.array(df["t_select"])
		at_train[pr, qb] += np.array(df["t_train"])
		at_compress[pr, qb] += np.array(df["t_compress"])
		at_decode[pr, qb] += np.array(df["t_decode"])
		at_aggregate[pr, qb] += np.array(df["t_aggregate"])
		at_round[pr, qb] += np.array(df["t_round"])

steps = np.array([i for i in range(1,101)])


# Baseline
print("---------------------------")
print("-------  BASELINE  --------")
print("---------------------------")
print("")
print("Results:")
print("Final accuracy: ", accuracy[0,-1]/3)
print("Final loss:     ", train_loss[0,-1]/3)
print("Avg model size: ", np.mean(size[0]/3))
print("Avg t_select:   ", np.mean(t_select[0]/3))
print("Avg t_train:    ", np.mean(t_train[0]/3))
print("Avg t_compress: ", np.mean(t_compress[0]/3))
print("Avg t_decode:   ", np.mean(t_decode[0]/3))
print("Avg t_aggregate:", np.mean(t_aggregate[0]/3))
print("Avg t_round:    ", np.mean(t_round[0]/3))

min_, max_ = min(accuracy[0]), max(accuracy[0])

# Own method to calculate the model
K = max_ - min_
tau = steps[np.argmax(accuracy[0] > 0.63*K+min_)]
bias = min_

# Calculate the model and evaluate it
accuracy_calc = first_order_response(steps, K, tau, bias)
r2 = 1-np.sum((accuracy[0] - accuracy_calc)**2)/np.sum((accuracy[0] - np.mean(accuracy[0]))**2)

print("tau:            ", tau)
print("r2:             ", r2)
print("")

for pr in range(10):
	for qb in range(2):
		size = sizes[pr, qb] /3
		accuracy = accuracies[pr, qb] /3
		train_loss = train_losses[pr, qb] /3
		t_select = at_select[pr, qb] /3
		t_train = at_train[pr, qb] /3
		t_compress = at_compress[pr, qb] /3
		t_decode = at_decode[pr, qb] /3
		t_decode = at_aggregate[pr, qb] /3
		t_round = at_round[pr, qb] /3

		min_, max_ = min(accuracy), max(accuracy)

		# Own method to calculate the model
		K = max_ - min_
		tau = steps[np.argmax(accuracy > 0.63*K+min_)]
		bias = min_

		# Calculate the model and evaluate it
		accuracy_calc = first_order_response(steps, K, tau, bias)
		r2 = 1-np.sum((accuracy - accuracy_calc)**2)/np.sum((accuracy - np.mean(accuracy))**2)

		print("---------------------------")
		print("SETTINGS")
		print("Pruning:       ", 0.95 if pr==0 else 0.1*pr)
		print("Quantization:  ", qb*4 + 4)
		print("")
		print("RESULTS")
		print("Final accuracy: ", accuracy[-1])
		print("Final loss:     ", train_loss[-1])
		print("")
		print("Avg model size: ", np.mean(size))
		print("")
		print("Avg t_select:   ", np.mean(t_select))
		print("Avg t_train:    ", np.mean(t_train))
		print("Avg t_compress: ", np.mean(t_compress))
		print("Avg t_decode:   ", np.mean(t_decode))
		print("Avg t_aggregate:", np.mean(t_aggregate))
		print("Avg t_round:    ", np.mean(t_round))
		print("")
		print("tau:            ", tau)
		print("r2:             ", r2)
		print("")
