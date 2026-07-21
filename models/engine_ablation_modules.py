import torch
import torch.nn as nn

class AblatableUMSGArchitecture(nn.Module):
    """
    PyTorch Implementation for Table 8 Ablation Experiments.
    Allows toggling Semantic, Spatial, and Structural Engines independently.
    """
    def __init__(self, use_semantic=True, use_spatial=True, use_structural=True, embed_dim=768, k_shots=0):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_spatial = use_spatial
        self.use_structural = use_structural
        self.k_shots = k_shots
        self.embed_dim = embed_dim

        # Conditional Module Initialization
        if self.use_semantic:
            self.semantic_engine = nn.Linear(1024, embed_dim)
        if self.use_spatial:
            self.spatial_engine = nn.Linear(3, embed_dim)
        if self.use_structural:
            self.structural_engine = nn.Linear(3, embed_dim)

        # Context Adaptation Head for Few-Shot (k=4)
        if self.k_shots > 0:
            self.shot_adaptor = nn.Sequential(
                nn.Linear(embed_dim * k_shots, embed_dim),
                nn.GELU()
            )

        # Multi-Modal Cross Fusion Core
        self.fusion_core = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, batch_first=True)
        self.text_decoder = nn.Linear(embed_dim, 1000)

    def forward(self, text_tokens=None, spatial_points=None, structural_points=None, support_shots=None):
        batch_size = 1
        device = next(self.parameters()).device

        # 1. Semantic Engine Features
        if self.use_semantic and text_tokens is not None:
            batch_size = text_tokens.size(0)
            sem_feat = self.semantic_engine(text_tokens)
        else:
            sem_feat = torch.zeros((batch_size, 10, self.embed_dim), device=device)

        # 2. Spatial Engine Features
        if self.use_spatial and spatial_points is not None:
            batch_size = spatial_points.size(0)
            spat_feat = self.spatial_engine(spatial_points)
        else:
            spat_feat = torch.zeros((batch_size, 100, self.embed_dim), device=device)

        # 3. Structural Engine Features
        if self.use_structural and structural_points is not None:
            batch_size = structural_points.size(0)
            struct_feat = self.structural_engine(structural_points)
        else:
            struct_feat = torch.zeros((batch_size, 100, self.embed_dim), device=device)

        # Combine Spatial & Structural Manifolds
        spat_struct_combined = spat_feat + struct_feat

        # Inject In-Context Few-Shot Exemplars
        if self.k_shots > 0 and support_shots is not None and self.use_spatial:
            shot_feats = self.spatial_engine(support_shots).view(batch_size, -1)
            boost = self.shot_adaptor(shot_feats).unsqueeze(1)
            spat_struct_combined = spat_struct_combined + boost

        # Fusion Core Pass
        fused, _ = self.fusion_core(spat_struct_combined, sem_feat, sem_feat)
        logits = self.text_decoder(fused.mean(dim=1))
        return logits