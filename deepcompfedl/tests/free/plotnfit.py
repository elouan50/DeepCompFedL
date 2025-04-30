"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/home/colybes/Downloads/wandb_export_2025-04-11T09_48_32.955+02_00.csv"
df = pd.read_csv(file_path)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau))+bias

# Extract data
steps = df['Step']
accuracy = df['p0.3-q8-e1-n3 - accuracy']

# Adjust to fit the csv data
min_, max_ = min(accuracy), accuracy.iloc[-1]

# Own method to calculate the model
K_hand = max_ - min_
tau_hand = steps[np.argmax(accuracy > 0.63*K_hand+min_)] 
bias_hand = min_

# Calculate the model and evaluate it
accuracy_hand = first_order_response(steps, K_hand, tau_hand, bias_hand)
mse_hand = np.mean((accuracy - accuracy_hand) ** 2)
r2_hand = 1-np.sum((accuracy-accuracy_hand)**2)/np.sum((accuracy-np.mean(accuracy))**2)

# Plot the accuracy values
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracy, label='Accuracy', color='b')
plt.plot(steps, accuracy_hand, label='Calculated model', color='r')

# Labels and title
plt.xlabel('Step')
plt.xlim(0, 100)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Training Accuracy Across Steps')
plt.legend()
plt.grid()
plt.figtext(0.4, 0.25, f"τ = {int(tau_hand)}, C = {K_hand:.3f}, l = {bias_hand:.3f}", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
plt.figtext(0.4, 0.15, f"MSE = {mse_hand:.4f}\nR² = {r2_hand:.4f}", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})

# Save the plot
plt.savefig("/home/colybes/Downloads/plot.png")
