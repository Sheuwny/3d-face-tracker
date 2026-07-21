import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# PARADIGM 1: Multi-View / Voxel Paradigm
# Spatial: Discretized Voxel (3D Conv) | Fusion: Late Fusion
# =====================================================================
class MultiViewVoxelParadigm(nn.Module):
    def __init__(self, voxel_res=32, in_channels=1, text_dim=512, hidden_dim=256):
        super().__init__()
        self.voxel_res = voxel_res
        
        # 3D Convolutional Encoder for Cubic Voxel Grids O(V^3)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2), # Res: 16x16x16
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2), # Res: 8x8x8
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)) # Global feature vector
        )
        
        # Separate Late Fusion Layer
        self.text_proj = nn.Linear(text_dim, 128)
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Voxel probability score
        )

    def forward(self, voxel_grid, text_embeds):
        # voxel_grid shape: (Batch, 1, V, V, V) e.g., (B, 1, 32, 32, 32)
        spatial_feats = self.encoder(voxel_grid).squeeze(-1).squeeze(-1).squeeze(-1) # (B, 128)
        text_feats = self.text_proj(text_embeds) # (B, 128)
        
        # Late Fusion
        combined = torch.cat([spatial_feats, text_feats], dim=-1)
        out = self.fusion_head(combined)
        return out


# =====================================================================
# PARADIGM 2: Traditional 3D VLM Paradigm
# Spatial: Continuous Point Cloud | Primitive: Coarse 3D Bounding Boxes
# =====================================================================
class Traditional3DVLMParadigm(nn.Module):
    def __init__(self, num_anchors=9, text_dim=512, hidden_dim=256):
        super().__init__()
        self.num_anchors = num_anchors
        
        # PointNet-style spatial backbone
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Bounding Box Anchor Regressor (Center x,y,z + Size dX,dY,dZ)
        self.anchor_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_anchors * 6) # 6 parameters per box
        )
        
        # Late Stage Language Matcher
        self.text_matcher = nn.Linear(text_dim, 256)

    def forward(self, point_cloud, text_embeds):
        # point_cloud shape: (Batch, N, 3)
        B, N, _ = point_cloud.shape
        
        # Reshape for BatchNorm1d
        x = point_cloud.view(-1, 3)
        x = self.mlp1(x)
        x = x.view(B, N, 256)
        
        # Global max pooling across points
        global_spatial, _ = torch.max(x, dim=1) # (B, 256)
        
        # Predict bounding boxes
        pred_boxes = self.anchor_head(global_spatial).view(B, self.num_anchors, 6)
        
        # Anchor-based late alignment
        text_feat = self.text_matcher(text_embeds) # (B, 256)
        alignment_scores = torch.bmm(pred_boxes.view(B, self.num_anchors, 6), 
                                     text_feat.unsqueeze(-1)[:, :6, :]).squeeze(-1)
        
        return pred_boxes, alignment_scores


# =====================================================================
# PARADIGM 3: Point Transformer Paradigm
# Spatial: Local Neighborhood Manifolds | Language: Isolated
# =====================================================================
class PointTransformerParadigm(nn.Module):
    def __init__(self, k_neighbors=16, in_dim=3, hidden_dim=128):
        super().__init__()
        self.k = k_neighbors
        self.fc_delta = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.w_qs = nn.Linear(in_dim, hidden_dim)
        self.w_ks = nn.Linear(in_dim, hidden_dim)
        self.w_vs = nn.Linear(in_dim, hidden_dim)

    def _knn(self, x, k):
        # Simple Euclidean Distance KNN calculation
        dist = torch.cdist(x, x) # (B, N, N)
        _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
        return idx

    def forward(self, point_cloud):
        # point_cloud shape: (B, N, 3)
        B, N, _ = point_cloud.shape
        k_idx = self._knn(point_cloud, self.k) # (B, N, K)
        
        # Extract local K-NN neighborhood coordinates
        idx_base = torch.arange(0, B, device=point_cloud.device).view(-1, 1, 1) * N
        flat_idx = (k_idx + idx_base).view(-1)
        flat_pts = point_cloud.view(-1, 3)[flat_idx].view(B, N, self.k, 3)
        
        # Local position encoding delta
        pos_delta = flat_pts - point_cloud.unsqueeze(2) # (B, N, K, 3)
        pos_enc = self.fc_delta(pos_delta) # (B, N, K, hidden_dim)
        
        # Query, Key, Value Projections
        q = self.w_qs(point_cloud).unsqueeze(2) # (B, N, 1, hidden_dim)
        k_flat = point_cloud.view(-1, 3)[flat_idx].view(B, N, self.k, 3)
        k = self.w_ks(k_flat) # (B, N, K, hidden_dim)
        v = self.w_vs(k_flat) # (B, N, K, hidden_dim)
        
        # Point Attention Weighting (Language Isolated)
        attn = self.fc_gamma(q - k + pos_enc)
        attn = F.softmax(attn, dim=2)
        
        out = torch.sum(attn * (v + pos_enc), dim=2) # (B, N, hidden_dim)
        return out


# =====================================================================
# PARADIGM 4: UMSG Network (Ours)
# Spatial: Unified Parallel Manifolds | Primitive: Anchor-Free Probability Field
# Fusion: Native Hidden Core Fusion
# =====================================================================
class UMSGNetworkParadigm(nn.Module):
    def __init__(self, spatial_dim=128, text_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        
        # Spatial Feature Extractor
        self.spatial_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, spatial_dim)
        )
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.spatial_embed = nn.Linear(spatial_dim, hidden_dim)
        
        # Native Hidden Core Cross-Modal Attention Engine
        self.cross_attention_core = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Anchor-Free Continuous Probability Field Decoder
        self.prob_field_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # Continuous Probability Score per point
            nn.Sigmoid()
        )

    def forward(self, point_cloud, text_tokens):
        # point_cloud: (B, N, 3), text_tokens: (B, M, text_dim)
        B, N, _ = point_cloud.shape
        
        # 1. Structural spatial embedding
        spatial_feats = self.spatial_proj(point_cloud) # (B, N, spatial_dim)
        spatial_tokens = self.spatial_embed(spatial_feats) # (B, N, hidden_dim)
        
        # 2. Semantic text projection
        semantic_tokens = self.text_proj(text_tokens) # (B, M, hidden_dim)
        
        # 3. Native Hidden Core Fusion (Layer 1 Interlock)
        fused_tokens, _ = self.cross_attention_core(
            query=spatial_tokens,
            key=semantic_tokens,
            value=semantic_tokens
        )
        fused_tokens = self.norm(spatial_tokens + fused_tokens)
        
        # 4. Continuous Anchor-Free Field Output
        probability_field = self.prob_field_head(fused_tokens) # (B, N, 1)
        return probability_field