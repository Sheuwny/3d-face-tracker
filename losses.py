import torch
import torch.nn as nn
import torch.nn.functional as F

class UMSGLossObjective(nn.Module):
    # Complete mathematical execution of the Multi-Task Loss Objective function.
    # Combines Vertex BCE, Coordinate Regression, and SO(3) Frobenius Norms.
    def __init__(self, lambda_1=1.0, lambda_2=0.5, lambda_3=1.0):
        super(UMSGLossObjective, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, pred_probs, true_masks, pred_coords, true_coords, pred_R, true_R):
        l_bce = F.binary_cross_entropy_with_logits(pred_probs.squeeze(-1), true_masks)
        l_mask = F.smooth_l1_loss(pred_coords, true_coords, reduction='mean')
        
        delta_R = pred_R - true_R
        l_pose = torch.sum(delta_R ** 2, dim=(1, 2)).mean()
        
        total_loss = (self.lambda_1 * l_bce) + (self.lambda_2 * l_pose) + (self.lambda_3 * l_mask)
        return total_loss, l_bce, l_pose, l_mask
