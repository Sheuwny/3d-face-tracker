import torch
from torch.utils.data import Dataset

class SpatialGroundingDataset(Dataset):
    def __init__(self, num_samples=64, num_points=1024):
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self): 
        return self.num_samples

    def __getitem__(self, idx):
        point_cloud = torch.randn(self.num_points, 3)
        text_features = torch.randn(20, 1024)
        true_masks = torch.randint(0, 2, (32,)).float()
        true_coords = torch.tensor([0.42, 0.18, 0.76, 0.1, 0.1, 0.2])
        true_R = torch.eye(3)
        return point_cloud, text_features, true_masks, true_coords, true_R
