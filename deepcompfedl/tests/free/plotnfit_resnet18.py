"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import wandb

api = wandb.Api()
project = "elouan50-rwth-aachen-university/deepcompfedl-exp1-resnet18"

accuracies = np.zeros((12,999))

runs = api.runs(project)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

for run in runs:
	df = run.history(keys=["accuracy"], samples=999)
	print(df.size, run.name)
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
	accuracies[pr] += np.array(df['accuracy'])

steps = np.array([i for i in range(1,1000)])

for pr in range(12):
	accuracy = accuracies[pr] / 3

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
	print("Pruning:       ", pr*0.1 if pr<10 else round(pr*0.04+0.55,2))
	print("")
	print("Results:")
	print("Final accuracy: ", accuracy[-1])
	print("tau:            ", tau)
	print("r2:  ", r2)
	print("")
