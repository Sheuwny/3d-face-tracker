import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. ANCHOR-BASED BASELINE (Zero-Shot Grid Anchor Tracking Head)
# =====================================================================
class AnchorBasedBaseline(nn.Module):
    def __init__(self, num_anchors=64, embed_dim=256):
        super().__init__()
        self.setting = "Zero-Shot"
        self.anchor_grid = nn.Parameter(torch.randn(num_anchors, 3))
        self.feature_encoder = nn.Linear(3, embed_dim)
        self.pose_head = nn.Linear(embed_dim, 3) # Roll, Pitch, Yaw orientation
        self.path_planner = nn.Linear(embed_dim, 100) # Predicted trajectory waypoints
        self.cider_head = nn.Linear(embed_dim, 1000)

    def forward(self, point_cloud):
        # Input: (Batch, Num_Points, 3)
        pts_feat = self.feature_encoder(point_cloud).mean(dim=1)
        predicted_pose = self.pose_head(pts_feat)
        path_waypoints = self.path_planner(pts_feat)
        cider_logits = self.cider_head(pts_feat)
        return predicted_pose, path_waypoints, cider_logits


# =====================================================================
# 2. PROBABILITY TRACKING HEAD (Supervised Density Tracking)
# =====================================================================
class ProbabilityTrackingHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.setting = "Fully Supervised (No Zero-Shot)"
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.density_estimator = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # Mean pose vector
        )
        self.variance_estimator = nn.Linear(embed_dim, 3)
        self.path_head = nn.Linear(embed_dim, 100)
        self.cider_head = nn.Linear(embed_dim, 1000)

    def forward(self, point_cloud):
        feat = self.point_encoder(point_cloud).max(dim=1)[0]
        pose_mean = self.density_estimator(feat)
        pose_var = F.softplus(self.variance_estimator(feat))
        path_waypoints = self.path_head(feat)
        cider_logits = self.cider_head(feat)
        return (pose_mean, pose_var), path_waypoints, cider_logits


# =====================================================================
# 3. UMSG TRACKING HEAD (Ours - Zero-Shot Unified Manifold Spatial Head)
# =====================================================================
class UMSGTrackingHead(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.setting = "Zero-Shot"
        
        # Spatial & Structural Parallel Graph Manifolds
        self.spatial_manifold = nn.Linear(3, embed_dim)
        self.structural_manifold = nn.Linear(3, embed_dim)
        
        # Cross-Attentive Robotic Planning Core
        self.planning_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, batch_first=True)
        
        # Precision Robotic Heads
        self.precision_pose_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3) # Angular rotation prediction (Degrees)
        )
        self.path_execution_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 100) # Trajectory Path Waypoints
        )
        self.cider_gen_head = nn.Linear(embed_dim, 1000)

    def forward(self, point_cloud):
        spat = self.spatial_manifold(point_cloud)
        struct = self.structural_manifold(point_cloud)
        manifold = spat + struct
        
        fused_manifold, _ = self.planning_attention(manifold, manifold, manifold)
        pooled = fused_manifold.mean(dim=1)
        
        pose_deg = self.precision_pose_head(pooled)
        path_sr_waypoints = self.path_execution_head(pooled)
        cider_logits = self.cider_gen_head(pooled)
        
        return pose_deg, path_sr_waypoints, cider_logits