"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the CSV file
file_path = "/home/colybes/Downloads/wandb_export_2025-02-20T14_16_35.750+01_00.csv"
df = pd.read_csv(file_path)

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau))+bias

# Extract data
steps = df['Step']
accuracy = df['epochs: 10 - accuracy']
accuracy_min = df['epochs: 10 - accuracy__MIN']
accuracy_max = df['epochs: 10 - accuracy__MAX']

# Adjust to fit the csv data
min_, max_ = min(accuracy), max(accuracy)
params, _ = curve_fit(first_order_response, steps, accuracy, p0=[max_-min_, steps[np.argmax(accuracy > 0.63*(max_-min_))], min_])
K_opt, tau_opt, bias_opt = params

# Own method to calculate the model
K_hand = max_ - min_
tau_hand = steps[np.argmax(accuracy > 0.63*K_hand+min_)] 
bias_hand = min_

# Calculate the model and evaluate it
accuracy_fit = first_order_response(steps, K_opt, tau_opt, bias_opt)
accuracy_hand = first_order_response(steps, K_hand, tau_hand, bias_hand)
mse = np.mean((accuracy - accuracy_fit) ** 2)
mse_hand = np.mean((accuracy - accuracy_hand) ** 2)
r2 = 1-np.sum((accuracy-accuracy_fit)**2)/np.sum((accuracy-np.mean(accuracy))**2)
r2_hand = 1-np.sum((accuracy-accuracy_hand)**2)/np.sum((accuracy-np.mean(accuracy))**2)

# Plot the accuracy values
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracy, label='Accuracy', color='b')
plt.fill_between(steps, accuracy_min, accuracy_max, color='b', alpha=0.2, label='Min-Max Range')
plt.plot(steps, accuracy_fit, label='Estimated model', color='r')
plt.plot(steps, accuracy_hand, label='Hand-made model', color='g')

# Labels and title
plt.xlabel('Step')
plt.xlim(0, 100)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Training Accuracy Over Steps')
plt.legend()
plt.grid()
plt.figtext(0.4, 0.25, f"MSE = {mse:.4f}\nR² = {r2:.4f}", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
plt.figtext(0.4, 0.15, f"MSE_hand = {mse_hand:.4f}\nR²_hand = {r2_hand:.4f}", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})

# Save the plot
plt.savefig("/home/colybes/Downloads/plot.png")
