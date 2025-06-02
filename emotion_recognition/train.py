# train.py
# This module contains the main function for training the model.
# It manages the training loop, validation, metric calculation,
# and saving of the best model.

import torch
import torch.nn as nn
from tqdm import tqdm # To display a progress bar during training
import os
from utils import save_checkpoint # Import the function to save checkpoints

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir):
    """
    Main function for training a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): Optimizer (e.g., Adam).
        num_epochs (int): Number of epochs for training.
        device (torch.device): Device on which to run the training ('cuda' or 'cpu').
        checkpoint_dir (str): Directory where model checkpoints will be saved.

    Returns:
        tuple: The trained model and a dictionary containing the training history
        (loss and accuracy for training and validation).
    """
    best_val_accuracy = 0.0 # Tracks the best validation accuracy achieved
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    } # Dictionary to save training metrics

    # Training loop for the specified number of epochs
    for epoch in range(num_epochs):
        # -------------------- Training Phase --------------------
        model.train() # Sets the model to training mode (enables dropout, batch norm, etc.)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # tqdm wraps the iterator to show a progress bar
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")):
            inputs, labels = inputs.to(device), labels.to(device) # Moves inputs and labels to the device

            # Reset optimizer gradients
            optimizer.zero_grad()

            # Forward pass: calculates the model's output
            outputs = model(inputs)
            # Calculates the loss
            loss = criterion(outputs, labels)

            # Backward pass: calculates gradients
            loss.backward()
            # Updates model weights
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Accumulates batch loss
            
            # Calculates batch accuracy
            _, predicted = torch.max(outputs.data, 1) # Gets the class with the highest probability
            total_train += labels.size(0) # Adds the number of samples in the batch to the total
            correct_train += (predicted == labels).sum().item() # Counts correct predictions

        epoch_train_loss = running_loss / len(train_loader.dataset) # Average loss for the epoch
        epoch_train_accuracy = 100 * correct_train / total_train # Accuracy for the epoch

        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)

        # -------------------- Validation Phase --------------------
        model.eval() # Sets the model to evaluation mode (disables dropout, batch norm, etc.)
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Disables gradient calculation for the validation phase
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100 * correct_val / total_val

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        # Prints epoch metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {epoch_train_loss:.4f}, Training Acc: {epoch_train_accuracy:.2f}%, '
              f'Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {epoch_val_accuracy:.2f}%')

        # Saves the best model based on validation accuracy
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            # Saves the state of the model, optimizer, epoch, and best accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
            }, filename=os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Checkpoint saved: Best validation accuracy: {best_val_accuracy:.2f}%")

    return model, history # Returns the trained model and the metrics history