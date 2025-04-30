"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import pandas as pd
import wandb

api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-femnist"

accuracies = np.zeros((6, 3, 2, 50))
training_times = np.zeros((6, 3, 2, 50))
compression_times = np.zeros((6, 3, 2, 50))
round_times = np.zeros((6, 3, 2, 50))
communication_overheads = np.zeros((6, 3, 2, 50))

# Define the mappings
pr_dic = {0: 0, 0.2: 1, 0.4: 2, 0.5: 3, 0.6: 4, 0.9: 5}
pr_dic_inv = {0: "0", 1: "0.2", 2: "0.4", 3: "0.5", 4: "0.6", 5: "0.9"}
qb_dic = {32: 0, 8: 1, 4: 2}
qb_dic_inv = {0: "32", 1: "8", 2: "4"}

runs = api.runs(project)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

for run in runs:
	if run.name[-1] == "1" or run.name[-1] == "2":
		# Get the history of the run
		c = run.config
		df = run.history(keys=[
							"accuracy",
							"training-time",
							"compression-time",
							"round-time",
							"communication-overhead"
							])
		print(df.size, run.name, c["full-compression"])


		# Extract data
		accuracies[
				pr_dic[c["pruning-rate"]],
				qb_dic[c["bits-quantization"]],
				int(c["full-compression"])
				] += np.array(df['accuracy'])
		training_times[
				pr_dic[c["pruning-rate"]],
				qb_dic[c["bits-quantization"]],
				int(c["full-compression"])
				] += np.array(df['training-time'])
		compression_times[
				pr_dic[c["pruning-rate"]],
				qb_dic[c["bits-quantization"]],
				int(c["full-compression"])
				] += np.array(df['compression-time'])
		round_times[
				pr_dic[c["pruning-rate"]],
				qb_dic[c["bits-quantization"]],
				int(c["full-compression"])
				] += np.array(df['round-time'])
		communication_overheads[
				pr_dic[c["pruning-rate"]],
				qb_dic[c["bits-quantization"]],
				int(c["full-compression"])
				] += np.array(df['communication-overhead'])

steps = np.array([i for i in range(1,51)])

for pr in range(6):
	for qb in range(3):
		for fc in range(2):
			accuracy = accuracies[pr, qb, fc] /2
			t_train = training_times[pr, qb, fc] /2
			t_compress = compression_times[pr, qb, fc] /2
			t_round = round_times[pr, qb, fc] /2
			co = communication_overheads[pr, qb, fc] /2

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
			print("Pruning:       ", pr_dic_inv[pr])
			print("Quantization:       ", qb_dic_inv[qb])
			print("Full compression: ", bool(fc))
			print("")
			print("Results:")
			print("Final accuracy: ", accuracy[-1])
			print("Avg t_train: ", np.mean(t_train))
			print("Avg t_compress: ", np.mean(t_compress))
			print("Avg t_round: ", np.mean(t_round))
			print("Avg t_compute: ", np.mean(t_round*co))
			print("tau:            ", tau)
			print("r2:  ", r2)
			print("")
