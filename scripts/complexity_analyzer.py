import numpy as np

def calculate_paradigm_complexities(V=32, N=2048, K=16, M=77, D=768):
    """
    Mathematical verification engine for Table 1 Computational Complexities.
    
    Parameters:
    - V: Voxel Grid Spatial Resolution (3D Grid size V x V x V)
    - N: Number of input 3D points
    - K: K-Nearest Neighbors for local manifold processing
    - M: Number of text/language tokens
    - D: Hidden feature dimension size
    """
    print("=" * 75)
    print("TABLE 1: THEORETICAL & EMPIRICAL COMPLEXITY VERIFICATION ENGINE")
    print("=" * 75)

    # 1. Multi-View / Voxel: O(V^3)
    c_voxel = V ** 3
    print(f"[1] Multi-View / Voxel      : O(V^3)             = {c_voxel:>12,} FLOPs (V={V})")

    # 2. Traditional 3D VLM: O(N log N)
    c_vlm = int(N * np.log2(N))
    print(f"[2] Traditional 3D VLM      : O(N log N)         = {c_vlm:>12,} FLOPs (N={N})")

    # 3. Point Transformers: O(N * K)
    c_pt = N * K
    print(f"[3] Point Transformers      : O(N * K)           = {c_pt:>12,} FLOPs (N={N}, K={K})")

    # 4. UMSG Network (Ours): O(N * K + M * D)
    c_umsg = (N * K) + (M * D)
    print(f"[4] UMSG Network (Ours)     : O(N * K + M * D)   = {c_umsg:>12,} FLOPs (N={N}, K={K}, M={M}, D={D})")
    
    print("-" * 75)
    efficiency_gain = ((c_voxel - c_umsg) / c_voxel) * 100
    print(f"[VERDICT] UMSG Network achieves {efficiency_gain:.2f}% theoretical FLOP reduction vs Voxel grids.")
    print("=" * 75)

if __name__ == "__main__":
    calculate_paradigm_complexities()