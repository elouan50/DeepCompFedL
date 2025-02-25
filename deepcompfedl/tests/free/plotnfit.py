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

# Calculate the model and evaluate it
accuracy_fit = first_order_response(steps, K_opt, tau_opt, bias_opt)
mse = np.mean((accuracy - accuracy_fit) ** 2)
r2 = 1-np.sum((accuracy-accuracy_fit)**2)/np.sum((accuracy-np.mean(accuracy))**2) 

# Plot the accuracy values
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracy, label='Accuracy', color='b')
plt.fill_between(steps, accuracy_min, accuracy_max, color='b', alpha=0.2, label='Min-Max Range')
plt.plot(steps, accuracy_fit, label='Estimated model', color='r')

# Labels and title
plt.xlabel('Step')
plt.xlim(0, 100)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Training Accuracy Over Steps')
plt.legend()
plt.grid()
plt.figtext(0.4, 0.2, f"MSE = {mse:.4f}\nR² = {r2:.4f}", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})

# Save the plot
plt.savefig("/home/colybes/Downloads/plot.png")
