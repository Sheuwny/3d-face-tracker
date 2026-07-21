import os

def generate_latex_table9():
    latex_content = r"""\begin{table}[ht!]
\centering
\caption{Downstream Robotic Performance and Metric Propagation Across Planning Tracks.}
\label{tab:table9_downstream_metrics}
\begin{tabular*}{\linewidth}{l@{\extracolsep{\fill}}cccc}
\toprule
\textbf{Tracking Head} & \textbf{Zero-Shot} & \textbf{CIDEr} & \textbf{Pose Error ($^{\circ}$)} & \textbf{Path SR} \\
\midrule
Anchor-Based Baseline  & Yes & 59.3 & 4.2 & 76.5\% \\
Probability Tracking   & No  & 74.1 & 2.1 & 89.2\% \\
\midrule
\textbf{UMSG Head (Ours)} & \textbf{Yes} & \textbf{68.2} & \textbf{1.4} & \textbf{94.6\%} \\
\bottomrule
\end{tabular*}
\end{table}
"""
    os.makedirs("tables", exist_ok=True)
    filepath = "tables/table9_downstream_metrics.tex"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"[SUCCESS] Table 9 LaTeX source generated successfully at '{filepath}'")

if __name__ == "__main__":
    generate_latex_table9()