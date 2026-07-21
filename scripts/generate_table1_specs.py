import os

def generate_table1():
    modules = [
        ("Semantic Engine", 12, 1024, 16, 0.10),
        ("Spatial Engine", 6, 768, 12, 0.10),
        ("Structural Engine", 8, 768, 12, 0.10),
        ("Cross Fusion Core", 4, 768, 12, 0.15),
        ("Coord Decoder Head", 3, 512, 8, 0.00),
    ]

    latex_str = """\\begin{table}[t]
\\centering
\\caption{Architectural Specifications of UMSG Modules}
\\label{tab:table1_module_configurations}
\\begin{tabular*}{\\linewidth}{l@{\\extracolsep{\\fill}}cccc}
\\toprule
\\textbf{Module Name} & \\textbf{Layers} & \\textbf{Dim} & \\textbf{Heads} & \\textbf{Dropout} \\\\
\\midrule\n"""

    for name, layers, dim, heads, dropout in modules:
        latex_str += f"{name:<20} & {layers:<6} & {dim:<5} & {heads:<5} & {dropout:<7.2f} \\\\\n"

    latex_str += """\\bottomrule
\\end{tabular*}
\\end{table}"""

    os.makedirs("tables", exist_ok=True)
    with open("tables/generated_table1.tex", "w") as f:
        f.write(latex_str)

    print("[SUCCESS] Table 1 generated and saved to 'tables/generated_table1.tex'")

if __name__ == "__main__":
    generate_table1()