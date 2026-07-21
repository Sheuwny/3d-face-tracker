import torch
import torch.nn as nn

# =====================================================================
# 1. SEMANTIC ENGINE
# Specs: 12 Layers | Dim: 1024 | Heads: 16 | Dropout: 0.10
# =====================================================================
class SemanticEngine(nn.Module):
    def __init__(self, layers=12, d_model=1024, nhead=16, dropout=0.10):
        super().__init__()
        self.layers_count = layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        # Input: (Batch, Text_Seq_Len, 1024)
        return self.encoder(x)


# =====================================================================
# 2. SPATIAL ENGINE
# Specs: 6 Layers | Dim: 768 | Heads: 12 | Dropout: 0.10
# =====================================================================
class SpatialEngine(nn.Module):
    def __init__(self, layers=6, d_model=768, nhead=12, dropout=0.10):
        super().__init__()
        self.layers_count = layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        # Input: (Batch, Num_Points, 768)
        return self.encoder(x)


# =====================================================================
# 3. STRUCTURAL ENGINE
# Specs: 8 Layers | Dim: 768 | Heads: 12 | Dropout: 0.10
# =====================================================================
class StructuralEngine(nn.Module):
    def __init__(self, layers=8, d_model=768, nhead=12, dropout=0.10):
        super().__init__()
        self.layers_count = layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        # Input: (Batch, Num_Points, 768)
        return self.encoder(x)


# =====================================================================
# 4. CROSS FUSION CORE
# Specs: 4 Layers | Dim: 768 | Heads: 12 | Dropout: 0.15
# =====================================================================
class CrossFusionCore(nn.Module):
    def __init__(self, layers=4, d_model=768, nhead=12, dropout=0.15):
        super().__init__()
        self.layers_count = layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)

    def forward(self, spatial_tokens, semantic_tokens):
        # Target (spatial): (B, N, 768), Memory (semantic): (B, M, 768)
        return self.fusion_decoder(tgt=spatial_tokens, memory=semantic_tokens)


# =====================================================================
# 5. COORD DECODER HEAD
# Specs: 3 Layers | Dim: 512 | Heads: 8 | Dropout: 0.00
# =====================================================================
class CoordDecoderHead(nn.Module):
    def __init__(self, layers=3, d_model=512, nhead=8, dropout=0.00):
        super().__init__()
        self.layers_count = layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.coord_proj = nn.Linear(512, 3) # Predicts (x, y, z) bounding coordinates

    def forward(self, fused_tokens):
        # Input: (Batch, Num_Points, 512)
        out = self.decoder(fused_tokens)
        coords = self.coord_proj(out)
        return coords


# =====================================================================
# COMPLETE UMSG PIPELINE ARCHITECTURE WRAPPER
# =====================================================================
class FullUMSGArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_engine = SemanticEngine()
        self.spatial_engine = SpatialEngine()
        self.structural_engine = StructuralEngine()
        self.cross_fusion_core = CrossFusionCore()
        
        self.sem_proj = nn.Linear(1024, 768)
        self.proj_to_decoder = nn.Linear(768, 512)
        self.coord_decoder_head = CoordDecoderHead()

    def forward(self, text_tokens, spatial_points, structural_points):
        sem = self.semantic_engine(text_tokens)              # (B, M, 1024)
        spat = self.spatial_engine(spatial_points)           # (B, N, 768)
        struct = self.structural_engine(structural_points)   # (B, N, 768)
        
        spatial_combined = spat + struct
        sem_768 = self.sem_proj(sem)                         # (B, M, 768)
        
        fused = self.cross_fusion_core(spatial_combined, sem_768) # (B, N, 768)
        fused_512 = self.proj_to_decoder(fused)                    # (B, N, 512)
        coords = self.coord_decoder_head(fused_512)                 # (B, N, 3)
        return coords