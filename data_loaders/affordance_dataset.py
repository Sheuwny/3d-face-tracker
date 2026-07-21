import torch
from torch.utils.data import Dataset
import numpy as np
import json

class ScanReferDataset(Dataset):
    """ PyTorch Dataset Loader for ScanRefer 3D Grounding """
    def __init__(self, json_path="data/scanrefer/ScanRefer_filtered.json"):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Dummy 1024 points 3D point cloud simulation
        point_cloud = torch.randn(1024, 3) 
        description = item["description"]
        return point_cloud, description

class AffordanceNetDataset(Dataset):
    """ PyTorch Dataset Loader for 3D-AffordanceNet """
    def __init__(self, npz_path="data/affordance_net/affordance_samples.npz"):
        loaded = np.load(npz_path)
        self.points = loaded["points"]
        self.labels = loaded["labels"]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return torch.tensor(self.points[idx]), torch.tensor(self.labels[idx])