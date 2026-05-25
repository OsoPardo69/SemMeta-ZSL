import torch
import numpy as np
from torch.utils.data import Dataset


class MalwareTabularDataset(Dataset):
    """Tabular dataset that bundles behavioral features with their text prototype targets."""

    def __init__(self, features: np.ndarray, labels: list, prototypes_dict: dict):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels
        self.prototypes = torch.stack([
            torch.tensor(prototypes_dict[lbl], dtype=torch.float32)
            for lbl in labels
        ])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.prototypes[idx]
