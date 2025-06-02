# utils.py
# This module contains utility functions for the project,
# such as saving and loading model checkpoints
# and visualizing training metrics.

import torch
import matplotlib.pyplot as plt
import os

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Saves the current state of the model and optimizer as a checkpoint.

    Args:
        state (dict): A dictionary containing the state to save (e.g., model_state_dict, optimizer_state_dict).
        filename (str): The path and name of the file where the checkpoint will be saved.
    """
    print(f"Saving checkpoint to: {filename}")
    torch.save(state, filename)

def load_checkpoint(model, optimizer=None, filename="checkpoint.pth.tar"):
    """
    Loads a saved checkpoint to resume training or for inference.

    Args:
        model (nn.Module): The PyTorch model to load the state into.
        optimizer (Optimizer, optional): The optimizer to load the state into. Default: None.
        filename (str): The path and name of the checkpoint file to load.

    Returns:
        tuple: The updated model, the updated optimizer (or None), the starting epoch
        and the best validation accuracy from the checkpoint.
    """
    print(f"Loading checkpoint from: {filename}")
    # Maps loaded tensors to CPU if a GPU is not available
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))

    # Loads the model's state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Loads the optimizer's state ONLY IF an optimizer was provided
    # and if the optimizer's state is present in the checkpoint
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")
    elif optimizer is not None and 'optimizer_state_dict' not in checkpoint:
        print("Warning: Optimizer was provided but no 'optimizer_state_dict' found in checkpoint.")


    # Use .get() to retrieve values more safely, providing a default if not present
    epoch = checkpoint.get('epoch', 0) # Defaults to 0 if 'epoch' is not present
    best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0) # Defaults to 0.0 if 'best_val_accuracy' is not present

    print(f"Checkpoint loaded. Resuming from epoch {epoch}, best validation accuracy: {best_val_accuracy:.2f}%")
    return model, optimizer, epoch, best_val_accuracy

def plot_metrics(history, plots_dir):
    """
    Generates and saves plots of training and validation loss and accuracy.

    Args:
        history (dict): Dictionary containing lists of 'train_loss', 'train_accuracy',
                        'val_loss', 'val_accuracy'.
        plots_dir (str): Directory where the plots will be saved.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
    plt.plot(epochs, history['train_accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Automatically adjusts subplots to prevent overlapping
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png')) # Saves the plot
    plt.close() # Closes the figure to free up memory
    print(f"Training metrics plots saved to {os.path.join(plots_dir, 'training_metrics.png')}")