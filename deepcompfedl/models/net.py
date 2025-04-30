"""DeepCompFedL: A Flower / PyTorch app.

This module defines a simple Convolutional Neural Network (CNN) model using PyTorch.

Classes:
    Net: A simple CNN model adapted from 'PyTorch: A 60 Minute Blitz'.

Net class:
    Methods:
        __init__(): Initializes the CNN model with two convolutional layers, two pooling layers, and three fully connected layers.
        forward(x): Defines the forward pass of the network. Takes an input tensor `x` and returns the output tensor after passing through the network.
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, input_shape: tuple = (3,32,32), num_classes: int = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
