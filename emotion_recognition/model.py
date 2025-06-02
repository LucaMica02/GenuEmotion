# model.py
# This module defines the architecture of the Convolutional Neural Network (CNN)
# using PyTorch.

import torch
import torch.nn as nn
import torch.nn.functional as F # For functions like ReLU, MaxPooling

class EmotionCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for facial emotion classification.
    Architecture with convolutional blocks followed by pooling and fully connected layers.
    """
    def __init__(self, num_classes=7):
        """
        Constructor for the EmotionCNN class.

        Args:
            num_classes (int): The number of emotion classes to predict (default 7 for FER-2013).
        """
        super(EmotionCNN, self).__init__() # Calls the constructor of the base class nn.Module

        # Convolutional Block 1
        # Input: 1 channel (grayscale image), Output: 64 channels
        # Kernel: 3x3, Padding: 1 (to maintain spatial dimension)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # Batch normalization to stabilize training
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dimension (e.g., 48x48 -> 24x24)
        self.dropout1 = nn.Dropout(0.25) # Dropout to prevent overfitting

        # Convolutional Block 2
        # Input: 64 channels, Output: 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dimension (e.g., 24x24 -> 12x12)
        self.dropout2 = nn.Dropout(0.25)

        # Convolutional Block 3
        # Input: 128 channels, Output: 256 channels
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dimension (e.g., 12x12 -> 6x6)
        self.dropout3 = nn.Dropout(0.25)

        # Fully Connected Layers (FC)
        # After convolutional blocks, the output is flattened and passed to dense layers.
        # The input size for the first FC layer depends on the final dimensions
        # of the last convolutional layer's output after pooling.
        self.fc1 = nn.Linear(256 * 6 * 6, 256) # First FC layer
        self.bn7 = nn.BatchNorm1d(256) # Batch normalization for FC layers
        self.dropout4 = nn.Dropout(0.5) # Higher dropout for FC layers

        self.fc2 = nn.Linear(256, num_classes) # Output layer, with one neuron for each emotion class

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Describes how data flows through the various layers.

        Args:
            x (torch.Tensor): The input tensor (image).

        Returns:
            torch.Tensor: The output tensor (logits for emotion classes).
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x))) # Convolution -> Batch Norm -> ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x) # Max Pooling
        x = self.dropout1(x) # Dropout

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flattening: Flattens the convolutional output into a 1D vector
        # x.size(0) is the batch size
        x = x.view(x.size(0), -1) # -1 automatically calculates the remaining dimension

        # Fully Connected Layers
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x) # Final output (logits)

        return x # Returns the logits, which will then be passed to a loss function (e.g., CrossEntropyLoss)