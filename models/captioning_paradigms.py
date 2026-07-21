import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. SCANREFER BASELINE (Fully Supervised PointNet++ + GRU Captioner)
# =====================================================================
class ScanReferBaseline(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=256):
        super().__init__()
        self.setting = "Fully Supervised"
        self.point_feature_extractor = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        self.caption_decoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, vocab_size)
        self.coord_head = nn.Linear(hidden_dim, 3)

    def forward(self, point_cloud):
        # Input: (Batch, Num_Points, 3)
        pts_feat = self.point_feature_extractor(point_cloud).mean(dim=1) # Global pooling
        coords = self.coord_head(pts_feat) # Predicted 3D Center (x, y, z)
        logits = self.text_head(pts_feat) # Token logits
        return coords, logits


# =====================================================================
# 2. 3D-LLM ENGINE (Zero-Shot k=0, Point Encoder + LLM Projection)
# =====================================================================
class ThreeDLLMEngine(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=1000):
        super().__init__()
        self.setting = "Zero-Shot (k=0)"
        self.encoder_3d = nn.Linear(3, embed_dim) # Fixed: Renamed from 3d_encoder to encoder_3d
        self.llm_proj = nn.Linear(embed_dim, embed_dim)
        self.text_head = nn.Linear(embed_dim, vocab_size)
        self.box_head = nn.Linear(embed_dim, 3)

    def forward(self, point_cloud):
        feat = F.relu(self.encoder_3d(point_cloud)).max(dim=1)[0]
        llm_embed = self.llm_proj(feat)
        coords = self.box_head(llm_embed)
        logits = self.text_head(llm_embed)
        return coords, logits


# =====================================================================
# 3. MULTI-MODAL GROUNDER (Zero-Shot k=0, Cross-Attention)
# =====================================================================
class MultiModalGrounder(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=1000):
        super().__init__()
        self.setting = "Zero-Shot (k=0)"
        self.point_encoder = nn.Linear(3, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.text_head = nn.Linear(embed_dim, vocab_size)
        self.coord_head = nn.Linear(embed_dim, 3)

    def forward(self, point_cloud, prompt_tokens=None):
        pts_feat = self.point_encoder(point_cloud)
        fused, _ = self.cross_attn(pts_feat, pts_feat, pts_feat)
        global_feat = fused.mean(dim=1)
        coords = self.coord_head(global_feat)
        logits = self.text_head(global_feat)
        return coords, logits


# =====================================================================
# 4. UMSG NETWORK (Zero-Shot k=0 & Few-Shot k=4 Flexible Paradigm)
# =====================================================================
class UMSGNetworkCaptioner(nn.Module):
    def __init__(self, embed_dim=768, vocab_size=1000, k_shots=0):
        super().__init__()
        self.k_shots = k_shots
        self.setting = f"Zero-Shot (k=0)" if k_shots == 0 else f"Few-Shot (k={k_shots})"
        
        # Unified Parallel Manifolds Engine
        self.spatial_manifold = nn.Linear(3, embed_dim)
        self.structural_manifold = nn.Linear(3, embed_dim)
        
        # Few-shot Context Adaptor Head
        if k_shots > 0:
            self.shot_adaptor = nn.Sequential(
                nn.Linear(embed_dim * k_shots, embed_dim),
                nn.GELU()
            )
        
        self.cross_fusion = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, batch_first=True)
        self.coord_decoder = nn.Linear(embed_dim, 3)
        self.caption_decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, point_cloud, support_shots=None):
        # Joint Manifold Representation
        spat = self.spatial_manifold(point_cloud)
        struct = self.structural_manifold(point_cloud)
        parallel_manifold = spat + struct
        
        if self.k_shots > 0 and support_shots is not None:
            # Incorporate k-shot exemplars into feature space
            shot_feats = self.spatial_manifold(support_shots).reshape(point_cloud.size(0), -1)
            context_boost = self.shot_adaptor(shot_feats).unsqueeze(1)
            parallel_manifold = parallel_manifold + context_boost

        fused, _ = self.cross_fusion(parallel_manifold, parallel_manifold, parallel_manifold)
        pooled = fused.mean(dim=1)
        
        pred_coords = self.coord_decoder(pooled)
        pred_logits = self.caption_decoder(pooled)
        return pred_coords, pred_logits