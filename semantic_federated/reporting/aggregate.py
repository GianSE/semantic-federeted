import os

import pandas as pd

GROUP_COLS = ["dataset", "latent_dim", "noise_level", "channel_type", "partition", "baseline_comm_mode"]
METRIC_COLS = ["accuracy_baseline", "accuracy_compressed", "classification_loss", "reconstruction_loss"]


def aggregate_over_seeds(results_csv: str, out_dir: str) -> pd.DataFrame:
    """Agrega resultados de múltiplas seeds em média +/- desvio padrão por configuração.

    Necessário porque uma única seed não é suficiente para distinguir um efeito real
    (ex.: ruído como regularizador) de variabilidade de inicialização.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    group_cols = [col for col in GROUP_COLS if col in df.columns]
    metric_cols = [col for col in METRIC_COLS if col in df.columns]

    agg_spec = {col: ["mean", "std", "count"] for col in metric_cols}
    summary = df.groupby(group_cols, dropna=False).agg(agg_spec)
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary = summary.reset_index()

    csv_path = os.path.join(out_dir, "results_summary.csv")
    summary.to_csv(csv_path, index=False)
    return summary


if __name__ == "__main__":
    aggregate_over_seeds("./results/data/experiment_results.csv", "./results/tables")
