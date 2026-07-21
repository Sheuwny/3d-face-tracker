\# Dataset Directory \& Setup Guide



This project benchmarks on three primary 3D vision and robotics datasets.



\## Directory Structure

Reviewers can reproduce our experiments by placing datasets in the following structure:

data/

├── scanrefer/

│   ├── ScanRefer\_filtered.json          # Text annotations

│   └── scans/                           # Point cloud PLY/NPY files from ScanNet

├── affordance\_net/

│   ├── affordance\_samples.npz           # Point cloud coordinates and segmentation

│   └── annotations.json                 # Action/Interaction semantics

└── robotic\_tracking/

└── trajectories.pth                 # Downstream robotic planning waypoints



\## Dataset Download Links

1\. \*\*ScanRefer\*\*: Download via \[Official ScanRefer Repo](https://github.com/daichiraun/ScanRefer).

2\. \*\*3D-AffordanceNet\*\*: Download via \[3D-AffordanceNet Official Source](https://github.com/3D-AffordanceNet/3D-AffordanceNet).

3\. \*\*Robotic Planning Benchmark\*\*: Generated via our custom simulation harness.

