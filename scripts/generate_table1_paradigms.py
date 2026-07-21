import os

def generate_latex_table1():
    """
    Automated exporter for Table 1: Structural and Computational Comparison
    Generates exact LaTeX source code directly into 'tables/table1_related_work_comparison.tex'
    """
    latex_content = r"""\begin{table*}[t!]
\centering
\caption{Structural and Computational Comparison of Spatial Grounding Paradigms}
\label{tab:table1_related_work_comparison}
\begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}cccc}
\toprule
\textbf{Paradigm} & \textbf{Spatial Representation} & \textbf{Localization Primitive} & \textbf{Computational Complexity} & \textbf{Cross-Modal Alignment} \\
\midrule
Multi-View / Voxel  & Discretized Voxel / 2D Projection & Fixed Grid / Pixel Voxels & $\mathcal{O}(V^3)$ (Cubic scaling) & Separate Late Fusion \\
Traditional 3D VLM  & Continuous Point Cloud Fields     & Coarse 3D Bounding Boxes  & $\mathcal{O}(N \log N)$           & Anchor-Based Late-Stage \\
Point Transformers  & Local Neighborhood Manifolds     & Vertex-Level Segmentations & $\mathcal{O}(N \cdot K)$          & Language Isolated \\
\midrule
\textbf{UMSG Network (Ours)} & \textbf{Unified Parallel Manifolds} & \textbf{Anchor-Free Probability Field} & $\mathbf{\mathcal{O}(N \cdot K + M \cdot D)}$ & \textbf{Native Hidden Core Fusion} \\
\bottomrule
\end{tabular*}
\end{table*}
"""
    os.makedirs("tables", exist_ok=True)
    filepath = "tables/table1_related_work_comparison.tex"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
    
    print(f"[SUCCESS] Table 1 LaTeX successfully written to '{filepath}'")

if __name__ == "__main__":
    generate_latex_table1()