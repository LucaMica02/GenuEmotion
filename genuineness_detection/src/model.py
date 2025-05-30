import torch.nn as nn

"""
Implements a feed-forward neural network (MLP) for binary classification of Action Units (AUs) using PyTorch.
The network consists of two fully connected layers, each followed by batch normalization, 
Leaky ReLU activation, and dropout for regularization to improve generalization and prevent overfitting,
While the final layer outputs a single logit for binary classification.
"""
# Define the feed-forward NN model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Define a sequential feed-forward neural network
        self.network = nn.Sequential(
            # Fully connected layer: input_dim -> 64 neurons
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            # Fully connected layer: 64 -> 32 neurons
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            # Final fully connected layer: 32 -> 1 neuron (output logit for binary classification)
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Pass the input through the network (all layers)
        # Output has shape (batch_size, 1); squeeze(1) removes the extra dimension
        # Final output shape: (batch_size,) — one logit per sample
        return self.network(x).squeeze(1)