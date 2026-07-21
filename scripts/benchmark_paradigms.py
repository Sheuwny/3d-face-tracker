import os
import sys
import time

# Parent directory (UMSG_Code root) ko Python path mein add kar rahe hain
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.paradigms_comparison import (
    MultiViewVoxelParadigm,
    PointTransformerParadigm,
    Traditional3DVLMParadigm,
    UMSGNetworkParadigm,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_paradigm_benchmark():
    batch_size = 2
    num_points = 1024
    voxel_res = 32
    text_len = 16
    text_dim = 512

    print("=" * 80)
    print("RUNNING LIVE SCRATCH EXECUTION FOR TABLE 1 PARADIGMS")
    print("=" * 80)

    # 1. Multi-View / Voxel
    model1 = MultiViewVoxelParadigm(voxel_res=voxel_res, text_dim=text_dim)
    dummy_voxels = torch.randn(batch_size, 1, voxel_res, voxel_res, voxel_res)
    dummy_text = torch.randn(batch_size, text_dim)

    start = time.time()
    out1 = model1(dummy_voxels, dummy_text)
    t1 = (time.time() - start) * 1000
    print(
        f"[1] Multi-View Voxel Paradigm   | Params: {count_params(model1):<10,} | Out Shape: {str(list(out1.shape)):<12} | Time: {t1:.2f} ms"
    )

    # 2. Traditional 3D VLM
    model2 = Traditional3DVLMParadigm(text_dim=text_dim)
    dummy_pc = torch.randn(batch_size, num_points, 3)

    start = time.time()
    boxes, align_scores = model2(dummy_pc, dummy_text)
    t2 = (time.time() - start) * 1000
    print(
        f"[2] Traditional 3D VLM Paradigm | Params: {count_params(model2):<10,} | Out Shape: {str(list(boxes.shape)):<12} | Time: {t2:.2f} ms"
    )

    # 3. Point Transformer
    model3 = PointTransformerParadigm(k_neighbors=16)

    start = time.time()
    out3 = model3(dummy_pc)
    t3 = (time.time() - start) * 1000
    print(
        f"[3] Point Transformer Paradigm  | Params: {count_params(model3):<10,} | Out Shape: {str(list(out3.shape)):<12} | Time: {t3:.2f} ms"
    )

    # 4. UMSG Network (Ours)
    model4 = UMSGNetworkParadigm(text_dim=text_dim)
    dummy_text_seq = torch.randn(batch_size, text_len, text_dim)

    start = time.time()
    prob_field = model4(dummy_pc, dummy_text_seq)
    t4 = (time.time() - start) * 1000
    print(
        f"[4] UMSG Network (Ours)         | Params: {count_params(model4):<10,} | Out Shape: {str(list(prob_field.shape)):<12} | Time: {t4:.2f} ms"
    )

    print("=" * 80)
    print(
        "[VERIFICATION PASSED] All paradigms fully operational and executable from scratch!"
    )


if __name__ == "__main__":
    run_paradigm_benchmark()