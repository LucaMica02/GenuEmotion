import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy
import os
from dataset import GenuinityDataset
from model import MLPClassifier
import matplotlib.pyplot as plt

# Read the best accuracy from the filepath
def read_accuracy(filepath):
    if (os.path.exists(filepath)):
        with open(filepath, 'r') as file:
            acc = file.read().strip()
            return float(acc)
    return 0.0
    
# Write the best accuracy to the filepath
def write_accuracy(filepath, acc):
    with open(filepath, 'w') as file:
        file.write(acc)

# Function to train and test the model classifier
def train_and_test(model, dataset_path, best_model_path, best_model_acc_path, k_folds, epochs):
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    metric = BinaryAccuracy().to(device)

    # Training loop
    def train_loop(dataloader, model, loss_fn, optimizer):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        model.train() # set model to training mode

        # Iterate over batches from the training dataloader
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Compute model outputs and loss and perform backpropagation
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            acc = metric(preds.int(), y_batch.int())
            metric.reset()

            total_correct += (preds.int() == y_batch.int()).sum().item()
            total_samples += y_batch.size(0)
            
            # Every 50 batches, print training loss and accuracy
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

        metric.reset()  # Reset accuracy metric for the next epoch
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    # Testing loop
    def test_loop(dataloader, model, loss_fn):
        test_loss = 0
        total_correct = 0
        total_samples = 0
        model.eval() # Set the model to evaluation mode

        # Initialize total test loss and containers for predictions and labels
        all_preds = []
        all_labels = []

        # Disable gradient computation for efficiency and memory savings during evaluation
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Perform forward pass, compute loss, generate binary predictions
                # and collect outputs and labels for evaluation
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                total_correct += (preds.int() == y_batch.int()).sum().item()
                total_samples += y_batch.size(0)
                all_preds.append(preds.int())
                all_labels.append(y_batch.int())
        
        # Aggregate predictions and labels, compute accuracy and average loss,
        # log results, and store fold accuracy
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = metric(all_preds, all_labels)
        metric.reset()
        avg_loss = test_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc.item():.4f}")
        fold_accuracies.append(acc.item())
        total_accuracies.append(acc.item())
        return avg_loss, avg_acc

    # Load data and preprocess
    df = pd.read_csv(dataset_path, delimiter=';')
    X = df.drop("class", axis=1).values
    y = df["class"].values

    # Set up cross-validation
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=31)
    
    tot_train_losses = [[] for _  in range(epochs)]
    tot_train_accuracies = [[] for _  in range(epochs)]
    tot_test_losses = [[] for _  in range(epochs)]
    tot_test_accuracies = [[] for _  in range(epochs)]
    total_accuracies = []
    best_accuracy = read_accuracy(best_model_acc_path)
    best_model_state = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        fold_accuracies = []

        # Split data for current fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create datasets and dataloaders
        train_dataset = GenuinityDataset(X_train, y_train)
        test_dataset = GenuinityDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Loss, optimizer, metric
        if os.path.exists(best_model_path): # Load weights from previous training if available
            model.load_state_dict(torch.load(best_model_path))
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # adds L2 penalty
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        metric = BinaryAccuracy().to(device)

        # Run training and testing
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer)
            test_loss, test_acc = test_loop(test_loader, model, loss_fn)
            tot_train_losses[epoch].append(train_loss)
            tot_train_accuracies[epoch].append(train_acc)
            tot_test_losses[epoch].append(test_loss)
            tot_test_accuracies[epoch].append(test_acc)
            scheduler.step()

        # Save best model
        fold_avg_acc = sum(fold_accuracies)/len(fold_accuracies)
        if fold_avg_acc > best_accuracy:
            best_accuracy = fold_avg_acc
            best_model_state = model.state_dict()
            print("New best model found and saved!")
        print(f"Average accuracy on the fold {fold+1}: {fold_avg_acc:.4f}")
        print("Training complete!")

    # Print average accuracy across folds
    if best_model_state:
        torch.save(best_model_state, best_model_path)
        write_accuracy(best_model_acc_path, str(best_accuracy))
    print(f"Average accuracy across {k_folds} folds: {sum(total_accuracies)/len(total_accuracies):.4f}")
    return {
        'train_loss': [sum(i) / len(i) for i in tot_train_losses],
        'train_accuracy': [sum(i) / len(i) for i in tot_train_accuracies],
        'val_loss': [sum(i) / len(i) for i in tot_test_losses],
        'val_accuracy': [sum(i) / len(i) for i in tot_test_accuracies]
    } 

# Train and test the model based on the Action Units
def train_au():
    PLOTS_DIR = 'plots'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 17
    model = MLPClassifier(input_size).to(device)
    dataset_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuineness_detection/dataset/AUs_dataset_shuffled.csv"
    best_model_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuineness_detection/au_classifier_best.pth"
    best_model_acc_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuineness_detection/au_accuracy.txt"
    k_folds = 4
    epochs = 40    
    history = train_and_test(model, dataset_path, best_model_path, best_model_acc_path, k_folds, epochs)
    #plot_metrics(history, PLOTS_DIR)

# Train and test the model based on the Landmarks
def train_landmarks():
    PLOTS_DIR = 'plots'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 936
    model = MLPClassifier(input_size).to(device)
    dataset_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuineness_detection/dataset/landmarks_dataset_shuffled.csv"
    best_model_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuineness_detection/land_classifier_best.pth"
    best_model_acc_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuineness_detection/land_accuracy.txt"
    k_folds = 4
    epochs = 40
    history = train_and_test(model, dataset_path, best_model_path, best_model_acc_path, k_folds, epochs)
    #plot_metrics(history, PLOTS_DIR)

if __name__ == "__main__":
    train_au()
    train_landmarks()