import torch
import torch.nn as nn

class MultiViewVoxelParadigm(nn.Module):
    """Paradigm 1: Discretized Voxel / 2D Projection with Late Fusion"""
    def __init__(self, voxel_res=32):
        super().__init__()
        self.voxel_res = voxel_res # O(V^3) Complexity
        self.voxel_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.late_fusion_linear = nn.Linear(32 + 512, 256)

    def forward(self, voxel_grid, text_feats):
        # voxel_grid shape: (B, 1, V, V, V)
        feat = self.voxel_encoder(voxel_grid)
        feat = feat.mean(dim=[2, 3, 4]) # Global pooling
        fused = self.late_fusion_linear(torch.cat([feat, text_feats], dim=-1))
        return fused

class Traditional3DVLMParadigm(nn.Module):
    """Paradigm 2: Continuous Point Clouds + Anchor-Based Late Fusion"""
    def __init__(self, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128))
        self.anchor_head = nn.Linear(128 + 512, 6) # Coarse Bounding Box (x,y,z,w,h,d)

    def forward(self, point_cloud, text_feats):
        # point_cloud: (B, N, 3)
        pc_feat = self.mlp(point_cloud).max(dim=1)[0]
        bbox_pred = self.anchor_head(torch.cat([pc_feat, text_feats], dim=-1))
        return bbox_pred

class PointTransformerParadigm(nn.Module):
    """Paradigm 3: Local Neighborhood Manifolds (Language Isolated)"""
    def __init__(self, k_neighbors=16):
        super().__init__()
        self.k = k_neighbors
        self.knn_conv = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 256))

    def forward(self, point_cloud):
        # Local K-NN processing without direct language interaction
        B, N, C = point_cloud.shape
        # Dummy representation of local manifold feature extraction
        features = self.knn_conv(torch.cat([point_cloud, point_cloud], dim=-1))
        return features

class UMSGNetworkParadigm(nn.Module):
    """Paradigm 4: Ours - Unified Parallel Manifolds with Native Hidden Core Fusion"""
    def __init__(self, hidden_dim=768, num_heads=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_modal_core = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.anchor_free_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1), # Probability field output
            nn.Sigmoid()
        )

    def forward(self, structural_tokens, semantic_tokens):
        # Native Hidden Core Fusion: O(N*K + (K+M)^2 * D)
        fused_attn, _ = self.cross_modal_core(query=structural_tokens, key=semantic_tokens, value=semantic_tokens)
        prob_field = self.anchor_free_prob_head(fused_attn)
        return prob_field