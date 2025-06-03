# main.py
# This file is the main entry point of the emotion recognition task.
# It handles environment setup, data loading, model initialization,
# starting the training process, and evaluating final performance.

'''COMMAND TO RUN THE CODE FROM TERMINAL: python main.py'''

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Import custom modules
from data_loader import FER2013Dataset # Module for data loading
from model import EmotionCNN # Module for model definition
from train import train_model # Module for the training function
from utils import save_checkpoint, load_checkpoint, plot_metrics # Module for utilities

# Define dataset and checkpoint paths
DATA_PATH = 'data/fer2013.csv'
CHECKPOINT_DIR = 'checkpoints'
PLOTS_DIR = 'plots'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Hyperparameters (Parameters that control the training process)
NUM_EPOCHS = 50 # Number of times the entire dataset will be passed through the network
BATCH_SIZE = 64 # Number of samples processed before updating model weights
LEARNING_RATE = 0.001 # Learning rate for the optimizer
NUM_CLASSES = 7 # Number of emotion classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

def main():
    # 1. Device Configuration (CPU/GPU)
    # Checks if a GPU (CUDA) is available and uses it; otherwise, uses the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define Image Transformations
    # These transformations are applied to images before being passed to the model.
    # They are essential for normalization and data augmentation.
    train_transforms = transforms.Compose([
        transforms.ToPILImage(), # Converts the NumPy array to a PIL Image
        transforms.Resize((48, 48)), # Resizes all images to 48x48 pixels (standard size for FER-2013)
        transforms.RandomHorizontalFlip(), # Applies a random horizontal flip (data augmentation)
        transforms.RandomRotation(10), # Applies a random rotation of +/- 10 degrees (data augmentation)
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Randomly changes brightness, contrast, saturation, hue
        transforms.ToTensor(), # Converts the PIL image to a PyTorch tensor (scales pixels from [0, 255] to [0, 1])
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizes pixel values to a range of [-1, 1]
    ])

    val_test_transforms = transforms.Compose([
        transforms.ToPILImage(), # Converts the NumPy array to a PIL Image
        transforms.Resize((48, 48)), # Resizes to 48x48 pixels
        transforms.ToTensor(), # Converts to tensor
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizes
    ])

    # 3. Dataset Loading
    # Initializes datasets for training, validation, and testing.
    # The FER2013Dataset will handle reading the CSV file and preparing the data.
    print("Loading FER-2013 dataset...")
    train_dataset = FER2013Dataset(csv_file=DATA_PATH, phase='Training', transform=train_transforms)
    val_dataset = FER2013Dataset(csv_file=DATA_PATH, phase='PublicTest', transform=val_test_transforms)
    test_dataset = FER2013Dataset(csv_file=DATA_PATH, phase='PrivateTest', transform=val_test_transforms)

    # Initialize DataLoaders
    # DataLoaders allow iterating over the dataset in batches and loading data in parallel.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Dataset loaded: Training={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # 4. Model Initialization
    # Creates an instance of our EmotionCNN neural network.
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device) # Moves the model to the device (GPU/CPU)
    print("Model initialized:")
    print(model)

    # 5. Define Loss Function and Optimizer
    # The loss function measures how far the model's predictions are from the truth.
    # The optimizer updates the model's weights to minimize the loss.
    criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer, popular for its effectiveness

    # 6. Model Training
    # Calls the training function defined in train.py.
    print("Starting model training...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        device,
        CHECKPOINT_DIR
    )
    print("Training completed.")

    # 7. Final Evaluation on Test Set
    # After training, evaluate the model's performance on the test set,
    # which was not used during training or validation.
    print("Final evaluation on Test Set...")
    model.eval() # Sets the model to evaluation mode (disables dropout, batch norm, etc.)
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): # Disables gradient calculation to save memory and speed up
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_dataset)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # 8. Plotting Training Metrics
    # Visualizes the trend of loss and accuracy during training.
    plot_metrics(history, PLOTS_DIR)
    print(f"Training plots saved to {PLOTS_DIR}")

if __name__ == '__main__':
    main()