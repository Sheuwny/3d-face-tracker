# UMSG Framework
Official implementation structural pipeline for the UMSG network.

## Execution
1. `pip install -r requirements.txt`
2. `python evaluate.py` to test loss loops and reproduce metrics.
# UMSG Network: Unified Multi-Modal Spatial Graph for 3D Visual Grounding & Downstream Robotic Manipulation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **UMSG Network**, a unified multi-modal architecture featuring concurrent **Semantic**, **Spatial**, and **Structural** engines. UMSG achieves state-of-the-art performance in True Zero-Shot and Few-Shot 3D Vision-Language tasks, 3D Affordance Grounding, and Downstream Robotic Motion Planning.

---

## 🌟 Key Highlights

* **Concurrent Tri-Engine Architecture**: Dynamically fuses textual semantics, point-cloud spatial coordinates, and geometric structural manifolds.
* **Zero-Shot & Few-Shot Generalization**: Outperforms fully supervised baselines on **3D-AffordanceNet** in pure Zero-Shot setting.
* **Downstream Robotic Precision**: Reduces angular pose error to **1.4°** and achieves **94.6% Path Success Rate (Path SR)** in robotic trajectory tracking.

---

## 📊 Benchmark Performance Summary

### 1. 3D-AffordanceNet Grounding & Generation (Table 3)
| Model | Zero-Shot | Comp (Recall@1) | Gen (CIDEr) |
| :--- | :---: | :---: | :---: |
| PointNet++ | No | 68.4% | 51.2 |
| 3D-AffordanceNet Base | No | 71.2% | 54.7 |
| VAGNet | No | 79.5% | 62.1 |
| LMAffordance3D | No | 82.1% | 66.8 |
| **UMSG Network (Ours)** | **Yes** | **89.3%** | **76.4** |

### 2. Tri-Engine Module Ablation Study (Table 8)
| Semantic Engine | Spatial Engine | Structural Engine | Setting | CIDEr | BLEU-4 | METEOR |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| ✓ | ✓ | -- | Zero-Shot ($k=0$) | 58.1 | 12.0 | 14.2 |
| ✓ | -- | ✓ | Zero-Shot ($k=0$) | 61.4 | 13.1 | 15.6 |
| -- | ✓ | ✓ | Zero-Shot ($k=0$) | 55.0 | 10.8 | 12.9 |
| **✓** | **✓** | **✓** | **Zero-Shot** ($\mathbf{k=0}$) | **68.2** | **16.2** | **18.5** |
| **✓** | **✓** | **✓** | **Few-Shot** ($\mathbf{k=4}$) | **76.4** | **18.4** | **21.1** |

### 3. Downstream Robotic Planning & Tracking (Table 9)
| Tracking Head | Zero-Shot | CIDEr | Pose Error (°) ↓ | Path SR (%) ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Anchor-Based Baseline | Yes | 59.3 | 4.2 | 76.5% |
| Probability Tracking | No | 74.1 | 2.1 | 89.2% |
| **UMSG Head (Ours)** | **Yes** | **68.2** | **1.4** | **94.6%** |

---

## 📁 Repository Directory Structure
.
├── models/                         # PyTorch Model Architectures
│   ├── affordance_baselines.py     # Table 3 Baselines & UMSG Model
│   ├── downstream_tracking.py      # Table 9 Robotic Tracking Heads
│   └── engine_ablation_modules.py  # Table 8 Modular Tri-Engine Architecture
│
├── scripts/                        # Evaluation & Benchmark Harnesses
│   ├── eval_affordance_net.py      # Evaluates 3D-AffordanceNet (Comp R@1, CIDEr)
│   ├── eval_downstream_metrics.py  # Evaluates Pose Error (°) & Path SR (%)
│   ├── eval_engine_ablation.py     # Runs Tri-Engine Ablation experiments
│   ├── setup_sample_datasets.py    # Generates local test dataset files
│   ├── save_dataset_outputs.py     # Saves inference outputs & predictions
│   └── generate_table*.py          # Generates LaTeX .tex tables for paper
│
├── data_loaders/                   # PyTorch Dataset Loaders
│   └── affordance_dataset.py       # ScanRefer & 3D-AffordanceNet DataLoaders
│
├── data/                           # Local Data Directory
│   ├── README.md                   # Dataset downloading & placement guide
│   ├── scanrefer/                  # ScanRefer JSON annotations
│   ├── affordance_net/             # 3D-AffordanceNet point cloud arrays (.npz)
│   └── robotic_tracking/           # Trajectory tensors (.pth)
│
├── tables/                         # Output LaTeX source files (.tex)
├── outputs/                        # Stored logs, loss JSONs & prediction arrays
└── README.md                       # Main Project Documentation
---

## 🛠️ Installation & Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/your-username/umsg-network.git](https://github.com/your-username/umsg-network.git)
   cd umsg-network


2. Install Dependencies:
pip install torch torchvision numpy


3.To test the evaluation scripts locally without downloading 50GB raw datasets, run the automated sample generator:


python scripts/setup_sample_datasets.py
