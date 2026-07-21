import os

def generate_latex_table2():
    latex_content = r"""\begin{table}[t]
\centering
\caption{Architectural Specifications of UMSG Modules}
\label{tab:table2_module_configurations}
\begin{tabular*}{\linewidth}{l@{\extracolsep{\fill}}cccc}
\toprule
\textbf{Module Name} & \textbf{Layers} & \textbf{Dim} & \textbf{Heads} & \textbf{Dropout} \\
\midrule
Semantic Engine   & 12 & 1024 & 16 & 0.10 \\
Spatial Engine    & 6  & 768  & 12 & 0.10 \\
Structural Engine & 8  & 768  & 12 & 0.10 \\
Cross Fusion Core & 4  & 768  & 12 & 0.15 \\
Coord Decoder Head& 3  & 512  & 8  & 0.00 \\
\bottomrule
\end{tabular*}
\end{table}
"""
    os.makedirs("tables", exist_ok=True)
    filepath = "tables/table2_module_configurations.tex"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
        
    print(f"[SUCCESS] Table 2 LaTeX code successfully generated at '{filepath}'")

if __name__ == "__main__":
    generate_latex_table2()