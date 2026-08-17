import os

import pandas as pd


COLUMNS = [
    "dataset",
    "model",
    "latent_dim",
    "snr_train_db",
    "snr_test_db",
    "channel",
    "num_clients",
    "rounds",
    "seed",
    "accuracy_baseline",
    "accuracy_compressed",
    "compression_ratio",
    "communication_cost_bits",
]


def generate_tables(results_csv: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    # Tolera colunas ausentes: a grade evolui entre fases do projeto.
    df = df[[c for c in COLUMNS if c in df.columns]]

    csv_path = os.path.join(out_dir, "results_table.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(out_dir, "results_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))

    print(df.to_string(index=False))


if __name__ == "__main__":
    generate_tables("./results/data/experiment_results.csv", "./results/tables")
