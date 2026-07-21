import os

def generate_latex_table8():
    latex_content = r"""\begin{table*}[t!]
\centering
\caption{Ablation Study of Concurrent Engine Modules evaluated across both True Zero-Shot ($k=0$) and Few-Shot ($k=4$) configurations.}
\label{tab:table8_engine_ablation}
\begin{tabular*}{\textwidth}{cccccc@{\extracolsep{\fill}}c}
\toprule
\textbf{Semantic Engine} & \textbf{Spatial Engine} & \textbf{Structural Engine} & \textbf{Setting} & \textbf{CIDEr} & \textbf{BLEU-4} & \textbf{METEOR} \\
\midrule
\checkmark & \checkmark & --          & Zero-Shot ($k=0$) & 58.1 & 12.0 & 14.2 \\
\checkmark & --         & \checkmark  & Zero-Shot ($k=0$) & 61.4 & 13.1 & 15.6 \\
--         & \checkmark & \checkmark  & Zero-Shot ($k=0$) & 55.0 & 10.8 & 12.9 \\
\midrule
\checkmark & \checkmark & \checkmark  & \textbf{Zero-Shot} ($\mathbf{k=0}$) & \textbf{68.2} & \textbf{16.2} & \textbf{18.5} \\
\checkmark & \checkmark & \checkmark  & \textbf{Few-Shot} ($\mathbf{k=4}$)  & \textbf{76.4} & \textbf{18.4} & \textbf{21.1} \\
\bottomrule
\end{tabular*}
\end{table*}
"""
    os.makedirs("tables", exist_ok=True)
    filepath = "tables/table8_engine_ablation.tex"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
        
    print(f"[SUCCESS] Table 8 LaTeX source exported to '{filepath}'")

if __name__ == "__main__":
    generate_latex_table8()