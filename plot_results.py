import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# Configurações para estilo IEEE/Acadêmico
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "figure.figsize": (5, 4),
    "savefig.dpi": 300
})

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_accuracy_vs_compression(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    compressed = df[df["accuracy_compressed"].notna()]
    if compressed.empty:
        return
    plt.figure()
    for dataset, group in compressed.groupby("dataset"):
        # Agrupar por compression_ratio para evitar linhas cruzadas se houver ruidos diferentes
        mean_group = group.groupby("compression_ratio")["accuracy_compressed"].mean().reset_index()
        plt.plot(mean_group["compression_ratio"], mean_group["accuracy_compressed"], marker="s", linestyle="--", label=f"{dataset} (avg)")
    plt.xlabel(r"Razão de Compressão (CR)")
    plt.ylabel(r"Acurácia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_compression_ratio.png"))
    plt.close()


def plot_accuracy_vs_latent_dim(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    compressed = df[df["accuracy_compressed"].notna()]
    if compressed.empty:
        return
    plt.figure()
    for dataset, group in compressed.groupby("dataset"):
        # Mostrar apenas noise=0.0 para clareza neste gráfico
        no_noise = group[group["noise_level"] == 0.0]
        plt.plot(no_noise["latent_dim"], no_noise["accuracy_compressed"], marker="o", label=f"{dataset} (No Noise)")
    plt.xlabel(r"Dimensão do Espaço Latente ($L$)")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_latent_dim.png"))
    plt.close()


def plot_comm_cost_vs_latent_dim(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    compressed = df[df["accuracy_compressed"].notna()]
    if compressed.empty:
        return
    plt.figure()
    for dataset, group in compressed.groupby("dataset"):
        # Pegar um valor único por latent_dim
        unique_bits = group.groupby("latent_dim")["communication_cost_bits"].first().reset_index()
        plt.semilogy(unique_bits["latent_dim"], unique_bits["communication_cost_bits"], marker="^", label=dataset)
    plt.xlabel(r"Dimensão do Espaço Latente ($L$)")
    plt.ylabel("Custo de Comunicação (bits)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "communication_cost_vs_latent_dim.png"))
    plt.close()


def plot_accuracy_vs_noise(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    compressed = df[df["accuracy_compressed"].notna()]
    if compressed.empty:
        return
    plt.figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, ((dataset, latent_dim), group) in enumerate(compressed.groupby(["dataset", "latent_dim"])):
        if i >= len(colors): i = 0
        plt.plot(group["noise_level"], group["accuracy_compressed"], marker="D", color=colors[i], label=fr"{dataset} $L={int(latent_dim)}$")
    plt.xlabel(r"Nível de Ruído ($\sigma$)")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.legend(ncol=1, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_noise_level.png"))
    plt.close()


def plot_accuracy_vs_round(history_csv: str, out_dir: str) -> None:
    """Curva de convergência federada: acurácia de teste por rodada."""
    if not os.path.isfile(history_csv):
        return
    hist = pd.read_csv(history_csv)
    if hist.empty or "eval_accuracy" not in hist.columns:
        return

    _ensure_dir(out_dir)
    for dataset, ds_group in hist.groupby("dataset"):
        plt.figure()
        for run_id, run_group in ds_group.groupby("run_id"):
            run_group = run_group.sort_values("round")
            row = run_group.iloc[0]
            if row.get("model") == "baseline":
                label = "baseline"
            else:
                label = fr"$L={int(row['latent_dim'])}$, $\sigma={row['noise_level']}$"
            plt.plot(run_group["round"], run_group["eval_accuracy"], marker="o", label=label)
        plt.xlabel("Rodada Federada ($t$)")
        plt.ylabel("Acurácia de Teste")
        plt.grid(True)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"accuracy_vs_round_{dataset}.png"))
        plt.close()


def generate_plots(data_dir: str, out_dir: str) -> None:
    df = pd.read_csv(os.path.join(data_dir, "experiment_results.csv"))
    # Ordenar por valores para o plot não ficar "vai e volta"
    df = df.sort_values(by=["dataset", "latent_dim", "noise_level"])
    plot_accuracy_vs_compression(df, out_dir)
    plot_accuracy_vs_latent_dim(df, out_dir)
    plot_comm_cost_vs_latent_dim(df, out_dir)
    plot_accuracy_vs_noise(df, out_dir)
    plot_accuracy_vs_round(os.path.join(data_dir, "history.csv"), out_dir)


if __name__ == "__main__":
    generate_plots("./results/data", "./results/plots")
