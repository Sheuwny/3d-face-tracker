import torch
import torch.nn as nn
import torch.nn.functional as F

class InContextShotScaler(nn.Module):
    """
    PyTorch Implementation of Dynamic In-Context Exemplar Scaler for Table 6.
    Processes variable context shots (k = 0, 1, 2, 4) to boost 3D spatial grounding.
    """
    def __init__(self, embed_dim=768, k_shots=0):
        super().__init__()
        self.k_shots = k_shots
        self.embed_dim = embed_dim
        
        # Unified Parallel Manifolds Encoders
        self.spatial_encoder = nn.Linear(3, embed_dim)
        self.structural_encoder = nn.Linear(3, embed_dim)
        
        # Flexible Shot Adaptor Module (Handles variable k exemplars)
        if k_shots > 0:
            self.exemplar_cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, 
                num_heads=12, 
                batch_first=True
            )
            self.shot_norm = nn.LayerNorm(embed_dim)
        
        # Heads for Grounding Precision & Caption Generation
        self.grounding_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # Predicts (x, y, z) spatial location
        )
        self.caption_head = nn.Linear(embed_dim, 1000)

    def forward(self, query_points, support_exemplars=None):
        """
        Args:
            query_points: (Batch, Num_Points, 3)
            support_exemplars: (Batch, k_shots, Num_Points, 3) if k_shots > 0 else None
        """
        # Extract query manifold features
        spat_feat = self.spatial_encoder(query_points)
        struct_feat = self.structural_encoder(query_points)
        query_manifold = spat_feat + struct_feat
        
        # Perform In-Context Adaptation if support exemplars (k > 0) are provided
        if self.k_shots > 0 and support_exemplars is not None:
            batch_size, k_count, num_pts, _ = support_exemplars.shape
            
            # Reshape exemplars to process through encoders
            flat_exemplars = support_exemplars.view(batch_size * k_count, num_pts, 3)
            exemplar_spat = self.spatial_encoder(flat_exemplars)
            exemplar_struct = self.structural_encoder(flat_exemplars)
            exemplar_feats = (exemplar_spat + exemplar_struct).mean(dim=1) # (B*k, D)
            exemplar_feats = exemplar_feats.view(batch_size, k_count, self.embed_dim) # (B, k, D)
            
            # Cross-attend query features with support exemplar context
            context_boost, _ = self.exemplar_cross_attn(
                query=query_manifold, 
                key=exemplar_feats, 
                value=exemplar_feats
            )
            query_manifold = self.shot_norm(query_manifold + context_boost)
        
        pooled_feature = query_manifold.mean(dim=1)
        pred_coords = self.grounding_head(pooled_feature)
        pred_caption_logits = self.caption_head(pooled_feature)
        
        return pred_coords, pred_caption_logits