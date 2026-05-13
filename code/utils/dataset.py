import torch
from torch.utils.data import Dataset
import numpy as np

class MTADDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)  # (样本数, 2, 100, 4)
        self.y = np.load(y_path)  # (样本数,)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        traj_a = torch.tensor(self.X[idx, 0], dtype=torch.float32)
        traj_b = torch.tensor(self.X[idx, 1], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return traj_a, traj_b, label