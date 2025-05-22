import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy
from torchvision import transforms

# 1. Dataset class for Images
class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = pd.read_csv(csv_file, delimiter=';')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        label = int(self.labels.iloc[index, 1])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label
    
# 2. Define the CNN model
class ClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 1 input channel (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # output single logit for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x.squeeze(1)  # remove extra dimension
    
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
        loss = loss_fn(outputs, y_batch.float())
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
            loss = loss_fn(outputs, y_batch.float())
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


# Setup
dataset_path = "C:/Users/lucam/Drive/Desktop/GenuEmotion/genuine_classification/dataset/images_dataset_shuffled.csv"
root_dir = "C:/Users/lucam/Drive/Desktop/dataset"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create the dataset
dataset = ImageDataset(root_dir, dataset_path, transform=transform)

# Extract labels
df = pd.read_csv(dataset_path, delimiter=';')
labels = df['class'].values

# Cross-validation
k_folds = 4
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=31)
epochs = 10
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df, labels)):
    #print(train_idx, test_idx)
    print(f"\n--- Fold {fold+1}/{k_folds} ---")

    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer, metric
    model = ClassifierCNN().to(device)
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