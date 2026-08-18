import os

import pandas as pd


COLUMNS = [
    "dataset",
    "model",
    "latent_dim",
    "latent_bits",
    "snr_train_db",
    "snr_test_db",
    "channel",
    "weight_snr_db",
    "rician_k_db",
    "beta",
    "num_clients",
    "rounds",
    "seed",
    "accuracy_baseline",
    "accuracy_compressed",
    # As duas contabilidades de comunicacao ficam explicitamente separadas:
    # payload de inferencia (onde a compressao semantica ganha) e troca de
    # pesos no FedAvg (onde o modelo semantico custa mais, por ter mais
    # parametros). Ver comm_cost.py.
    "inference_bits_per_sample",
    "compression_ratio",
    "bandwidth_savings",
    "model_params",
    "training_bits_total",
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
