from torch.utils.data import Dataset
import torch

# Dataset class
class GenuinityDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype('float32')
        self.labels = labels.astype('float32')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])