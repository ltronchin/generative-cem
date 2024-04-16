import torch
from torch.utils.data import Dataset

class Camelyon(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pass

