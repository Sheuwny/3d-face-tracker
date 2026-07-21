import os
import json
import numpy as np
import torch

print("=" * 60)
print("SETTING UP SAMPLE DATASET FILES ON YOUR LAPTOP")
print("=" * 60)

# 1. ScanRefer Sample JSON Data
scanrefer_data = [
    {
        "scene_id": "scene0000_00",
        "object_id": "0",
        "object_name": "chair",
        "description": "a black ergonomic office chair near the wooden desk"
    },
    {
        "scene_id": "scene0001_00",
        "object_id": "1",
        "object_name": "table",
        "description": "a large brown dining table in the center of the room"
    }
]

os.makedirs("data/scanrefer", exist_ok=True)
scanrefer_path = "data/scanrefer/ScanRefer_filtered.json"
with open(scanrefer_path, "w", encoding="utf-8") as f:
    json.dump(scanrefer_data, f, indent=4)
print(f"[CREATED] ScanRefer sample file: '{scanrefer_path}'")

# 2. 3D-AffordanceNet Point Cloud NumPy Data
os.makedirs("data/affordance_net", exist_ok=True)
points = np.random.rand(100, 1024, 3).astype(np.float32) # 100 sample point clouds
labels = np.random.randint(0, 10, size=(100, 1024))
affordance_path = "data/affordance_net/affordance_samples.npz"
np.savez(affordance_path, points=points, labels=labels)
print(f"[CREATED] 3D-AffordanceNet sample file: '{affordance_path}'")

# 3. Downstream Robotic Trajectories PyTorch Tensor
os.makedirs("data/robotic_tracking", exist_ok=True)
trajectories = {
    "waypoints": torch.randn(50, 100, 3), # 50 trajectories with 100 waypoints each
    "target_poses": torch.randn(50, 3)    # Target rotations
}
robotic_path = "data/robotic_tracking/trajectories.pth"
torch.save(trajectories, robotic_path)
print(f"[CREATED] Robotic Tracking sample file: '{robotic_path}'")

print("=" * 60)
print("[SUCCESS] All sample dataset files are ready on your laptop!")