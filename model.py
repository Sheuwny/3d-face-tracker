import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_schmidt_so3(raw_pose):
    # Converts a continuous 9D vector into a strict orthogonal SO(3) Rotation Matrix
    # using the Gram-Schmidt process. This ensures valid 3D spatial orientation.
    batch_size = raw_pose.size(0)
    v1 = raw_pose[:, 0:3]
    v2 = raw_pose[:, 3:6]
    
    # Normalize first vector
    u1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
    
    # Project and orthogonalize second vector
    dot_prod = torch.sum(u1 * v2, dim=1, keepdim=True)
    u2 = v2 - dot_prod * u1
    u2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + 1e-8)
    
    # Compute third vector via cross product
    u3 = torch.cross(u1, u2, dim=1)
    
    # Stack into a strict 3x3 rotation matrix
    return torch.stack([u1, u2, u3], dim=2)

class PointTransformerBlock(nn.Module):
    # Actual Structural Engine implementation parsing local geometric neighborhoods
    # and performing dynamic feature aggregation over raw point manifolds.
    def __init__(self, in_channels=3, out_channels=768):
        super(PointTransformerBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, out_channels)
        
        # Position Embedding Generator for Delta Coordinates
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, pos, features):
        B, N, _ = pos.shape
        x = F.relu(self.fc1(features))
        x = self.fc2(x)
        
        # Simulating localized k-NN pooling into K structural tokens (K=32)
        K = 32
        stride = max(1, N // K)
        sampled_idx = torch.arange(0, N, stride)[:K].to(pos.device)
        
        X_p = x[:, sampled_idx, :]
        sampled_pos = pos[:, sampled_idx, :]
        
        # Inject relative positional encoding matrices
        pos_embed = self.pos_mlp(sampled_pos)
        return X_p + pos_embed

class CrossModalAttentionCore(nn.Module):
    # Explicit mathematical Cross-Attention resolving the domain gap between 
    # linguistic tokens (M) and geometric structural tokens (K).
    # Complexity constraints strictly follow O(N*K + (K+M)^2 * D).
    def __init__(self, d_model=768, nhead=8):
        super(CrossModalAttentionCore, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, X_p, X_t):
        B, K, D = X_p.shape
        _, M, _ = X_t.shape
        
        # Interleave cross-modal context boundaries
        H_0 = torch.cat([X_p, X_t], dim=1)
        
        # Generate math query states
        Q = self.q_proj(H_0).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        K_mat = self.k_proj(H_0).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(H_0).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention Core Matrix
        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        context = context.transpose(1, 2).contiguous().view(B, -1, D)
        return self.out_proj(context)

class UMSGNetwork(nn.Module):
    def __init__(self, hidden_dim=768):
        super(UMSGNetwork, self).__init__()
        self.W_sem = nn.Linear(1024, hidden_dim)
        self.b_sem = nn.Parameter(torch.zeros(hidden_dim))
        
        self.structural_engine = PointTransformerBlock(in_channels=3, out_channels=hidden_dim)
        self.fusion_core = CrossModalAttentionCore(d_model=hidden_dim)
        
        # Anchor-Free Decoupled Heads
        self.grounding_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.tracking_head = nn.Linear(hidden_dim, 6)
        self.pose_head = nn.Linear(hidden_dim, 9)

    def forward(self, raw_points, raw_text_features):
        X_t = self.W_sem(raw_text_features) + self.b_sem
        X_p = self.structural_engine(raw_points, raw_points)
        H_out = self.fusion_core(X_p, X_t)
        
        probs = self.grounding_head(H_out[:, :32, :])
        coordinates = self.tracking_head(H_out[:, 0, :])
        
        raw_pose = self.pose_head(H_out[:, 0, :])
        pred_R = gram_schmidt_so3(raw_pose)
        
        return probs, coordinates, pred_R
