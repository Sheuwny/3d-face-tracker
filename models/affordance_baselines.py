import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. POINTNET++ BASELINE (Fully Supervised Point Cloud Backbone)
# =====================================================================
class PointNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.setting = "Fully Supervised"
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.affordance_head = nn.Linear(256, num_classes)
        self.cider_head = nn.Linear(256, 1)

    def forward(self, point_cloud):
        # Input: (Batch, Num_Points, 3)
        feat = F.relu(self.fc1(point_cloud))
        feat = F.relu(self.fc2(feat))
        feat = F.relu(self.fc3(feat))
        global_feat = torch.max(feat, dim=1)[0] # Global Max Pooling
        aff_logits = self.affordance_head(global_feat)
        cider_score = self.cider_head(global_feat)
        return aff_logits, cider_score


# =====================================================================
# 2. 3D-AFFORDANCENET BASE (Standard Point-Affordance Encoder)
# =====================================================================
class ThreeDAffordanceNetBase(nn.Module):
    def __init__(self, embed_dim=256, num_classes=10):
        super().__init__()
        self.setting = "Fully Supervised"
        self.backbone = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )
        self.affordance_cls = nn.Linear(embed_dim, num_classes)
        self.gen_head = nn.Linear(embed_dim, 512)

    def forward(self, point_cloud):
        feat = self.backbone(point_cloud).mean(dim=1)
        aff_logits = self.affordance_cls(feat)
        gen_out = self.gen_head(feat)
        return aff_logits, gen_out


# =====================================================================
# 3. VAGNET (Visual-Affordance Grounding Network)
# =====================================================================
class VAGNet(nn.Module):
    def __init__(self, embed_dim=512, num_classes=10):
        super().__init__()
        self.setting = "Fully Supervised"
        self.point_enc = nn.Linear(3, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.affordance_head = nn.Linear(embed_dim, num_classes)
        self.caption_proj = nn.Linear(embed_dim, 1000)

    def forward(self, point_cloud):
        pts = self.point_enc(point_cloud)
        attn_out, _ = self.attn(pts, pts, pts)
        pooled = attn_out.mean(dim=1)
        aff_logits = self.affordance_head(pooled)
        cap_logits = self.caption_proj(pooled)
        return aff_logits, cap_logits


# =====================================================================
# 4. LMAFFORDANCE3D (3D Language-Model Affordance Network)
# =====================================================================
class LMAffordance3D(nn.Module):
    def __init__(self, embed_dim=512, vocab_size=1000):
        super().__init__()
        self.setting = "Fully Supervised"
        self.encoder_3d = nn.Linear(3, embed_dim) # Proper variable naming
        self.lm_decoder = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.affordance_head = nn.Linear(embed_dim, 10)
        self.text_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, point_cloud):
        feat = F.relu(self.encoder_3d(point_cloud))
        dec_out = self.lm_decoder(feat, feat)
        pooled = dec_out.mean(dim=1)
        aff_logits = self.affordance_head(pooled)
        text_logits = self.text_head(pooled)
        return aff_logits, text_logits


# =====================================================================
# 5. UMSG NETWORK (Ours - Zero-Shot Parallel Manifolds Engine)
# =====================================================================
class UMSGNetworkAffordance(nn.Module):
    def __init__(self, embed_dim=768, vocab_size=1000):
        super().__init__()
        self.setting = "Zero-Shot"
        
        # Parallel Manifold Layers
        self.spatial_manifold = nn.Linear(3, embed_dim)
        self.structural_manifold = nn.Linear(3, embed_dim)
        
        self.fusion_core = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, batch_first=True)
        self.affordance_head = nn.Linear(embed_dim, 10)
        self.cider_gen_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, point_cloud, text_prompt=None):
        spat = self.spatial_manifold(point_cloud)
        struct = self.structural_manifold(point_cloud)
        joint_manifold = spat + struct
        
        fused, _ = self.fusion_core(joint_manifold, joint_manifold, joint_manifold)
        pooled = fused.mean(dim=1)
        
        aff_logits = self.affordance_head(pooled)
        gen_logits = self.cider_gen_head(pooled)
        return aff_logits, gen_logits