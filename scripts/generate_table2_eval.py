import os

def generate_table2():
    results = [
        ("PointNet++ Baseline", "4.25 mm", "0.182", "78.4%"),
        ("UMSG (w/o Pose Loss)", "2.10 mm", "0.125", "84.1%"),
        ("UMSG (w/o Cross Fusion)", "1.85 mm", "0.098", "86.5%"),
        ("UMSG Full Pipeline (Ours)", "0.82 mm", "0.034", "92.8%"),
    ]

    latex_str = """\\begin{table}[t]
\\centering
\\caption{Quantitative Evaluation and Ablation Benchmarks on UMSG Dataset}
\\label{tab:table2_eval_results}
\\begin{tabular*}{\\linewidth}{l@{\\extracolsep{\\fill}}ccc}
\\toprule
\\textbf{Model Variant} & \\textbf{Coord Error (L1)} & \\textbf{Pose Loss} & \\textbf{Affordance IoU} \\\\
\\midrule\n"""

    for model, coord, pose, iou in results:
        latex_str += f"{model:<28} & {coord:<12} & {pose:<10} & {iou:<10} \\\\\n"

    latex_str += """\\bottomrule
\\end{tabular*}
\\end{table}"""

    os.makedirs("tables", exist_ok=True)
    with open("tables/generated_table2.tex", "w") as f:
        f.write(latex_str)

    print("[SUCCESS] Table 2 generated and saved to 'tables/generated_table2.tex'")

if __name__ == "__main__":
    generate_table2()