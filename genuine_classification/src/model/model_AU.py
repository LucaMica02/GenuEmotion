import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy

# 1. Dataset class for AUs
class AUDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype('float32')
        self.labels = labels.astype('float32')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# 2. Define the feed-forward NN model
class AUClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x).squeeze(1)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
metric = BinaryAccuracy().to(device)

# Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            preds = torch.sigmoid(outputs) > 0.5
            acc = metric(preds.int(), y_batch.int())
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
    metric.reset()

# Testing loop
def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.int())
            all_labels.append(y_batch.int())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = metric(all_preds, all_labels)
    metric.reset()
    avg_loss = test_loss / len(dataloader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc.item():.4f}")
    fold_accuracies.append(acc.item())

# Load data and preprocess
dataset_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/AUs_dataset_shuffled.csv"
df = pd.read_csv(dataset_path, delimiter=';')

X = df.drop("class", axis=1).values
y = df["class"].values

# Set up cross-validation
k_folds = 4
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=31)
epochs = 10

fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    #print(train_idx, test_idx)
    print(f"\n--- Fold {fold+1}/{k_folds} ---")

    # Split data for current fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_dataset = AUDataset(X_train, y_train)
    test_dataset = AUDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer, metric
    model = AUClassifier(input_dim=X.shape[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metric = BinaryAccuracy().to(device)

    # Run training/testing
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Training complete!")

# Print average accuracy across folds
print(fold_accuracies)
print(f"Average accuracy across {k_folds} folds: {sum(fold_accuracies)/len(fold_accuracies):.4f}")