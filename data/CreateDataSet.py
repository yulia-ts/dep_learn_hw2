import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset

class VQA(Dataset):
    def __init__(self, n_features: int = 1024, n_samples: int = 1000):
        self.n_features = n_features
        self.n_samples = n_samples

        self.entries = self._create_entries()

    def _create_entries(self):
        entries = []

        for i in range(self.n_samples):
            entries.append({'x': torch.randn(self.n_features), 'y': 1})

        return entries

    def __getitem__(self, index):
        entry = self.entries[index]

        return entry['x'], entry['y']

    def __len__(self):
        return len(self.entries)
