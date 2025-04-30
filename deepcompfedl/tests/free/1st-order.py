"""
This file aims to test correlation between the steps and the accuracy
values of a neural network training process, and to fit a first-order
model to the data. The model is then evaluated and plotted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau))+bias



# Own method to calculate the model
steps = np.arange(0, 100, 1)
K_hand = 1
tau_hand = 10
bias_hand = 0

# Calculate the model and evaluate it
accuracy_hand = first_order_response(steps, K_hand, tau_hand, bias_hand)

# Plot the accuracy values
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracy_hand, label='1st order model', color='b')
# Add a vertical line at step=20
plt.axvline(x=tau_hand, ymax=0.63, color='g', linestyle='--')
plt.axhline(y=0.63, xmax=0.1, color='g', linestyle='--')
plt.axvline(x=3*tau_hand, ymax=0.95, color='r', linestyle='--')
plt.axhline(y=0.95, xmax=0.3, color='r', linestyle='--')
plt.axvline(x=5*tau_hand, ymax=0.99, color='y', linestyle='--')
plt.axhline(y=0.99, xmax=0.5, color='y', linestyle='--')

# Labels and title
plt.xlabel('Step')
plt.xlim(0, 100)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Step Response of a 1st Order Model')
plt.legend()
plt.grid()
plt.figtext(0.19, 0.07, f"t = τ", fontsize=10)
plt.figtext(0.34, 0.07, f"t = 3*τ", fontsize=10)
plt.figtext(0.5, 0.07, f"t = 5*τ", fontsize=10)

plt.figtext(0.09, 0.585, f"0.99\n0.95\n\n\n\n\n\n\n0.63", fontsize=10)
# Save the plot
plt.savefig("/home/colybes/Downloads/plot.png")
