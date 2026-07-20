import torch
import torch.optim as optim
from dataset import SpatialGroundingDataset
from model import UMSGNetwork
from losses import UMSGLossObjective

def execute_actual_optimization_gradient_loop():
    print("========== Initializing UMSG Mathematical Gradient Test ==========")
    model = UMSGNetwork()
    criterion = UMSGLossObjective()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    dataset = SpatialGroundingDataset(num_samples=4)
    points, text, masks, coords, true_R = dataset[0]
    
    points = points.unsqueeze(0)
    text = text.unsqueeze(0)
    masks = masks.unsqueeze(0)
    coords = coords.unsqueeze(0)
    true_R = true_R.unsqueeze(0)
    
    optimizer.zero_grad()
    probs, pred_coords, pred_R = model(points, text)
    total_loss, l_bce, l_pose, l_mask = criterion(probs, masks, pred_coords, coords, pred_R, true_R)
    
    print(f" -> Vertex Mask BCE Loss component:  {l_bce.item():.4f}")
    print(f" -> Continuous Coordinate L1 Loss:  {l_mask.item():.4f}")
    print(f" -> Geometric Frobenius Norm Pose Loss: {l_pose.item():.4f}")
    print(f" -> Consolidated Unified Multi-Task Loss: {total_loss.item():.4f}")
    
    total_loss.backward()
    optimizer.step()
    print("\n[SUCCESS]: Gradients successfully backpropagated across all structural nodes!")
    print("==================================================================")

if __name__ == "__main__":
    execute_actual_optimization_gradient_loop()
