import os

def generate_latex_table3():
    latex_content = r"""\begin{table}[ht!]
\centering
\caption{Comparative Evaluation on 3D-AffordanceNet}
\label{tab:table3_affordance_grounding}
\begin{tabular*}{\linewidth}{l@{\extracolsep{\fill}}ccc}
\toprule
\textbf{Model} & \textbf{Zero-Shot} & \textbf{Comp (R@1)} & \textbf{Gen (CIDEr)} \\
\midrule
PointNet++            & No  & 68.4\% & 51.2 \\
3D-AffordanceNet Base & No  & 71.2\% & 54.7 \\
VAGNet                & No  & 79.5\% & 62.1 \\
LMAffordance3D        & No  & 82.1\% & 66.8 \\
\midrule
\textbf{UMSG Network (Ours)} & Yes & \textbf{89.3\%} & \textbf{76.4} \\
\bottomrule
\end{tabular*}
\end{table}
"""
    os.makedirs("tables", exist_ok=True)
    filepath = "tables/table3_affordance_grounding.tex"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
        
    print(f"[SUCCESS] Table 3 LaTeX source file generated at '{filepath}'")

if __name__ == "__main__":
    generate_latex_table3()