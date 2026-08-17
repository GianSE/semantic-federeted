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
    plotted = False
    for dataset, group in compressed.groupby("dataset"):
        # Canal ideal (snr_train_db nulo) para isolar o efeito da compressão.
        ideal = group[group["snr_train_db"].isna()]
        if ideal.empty:
            continue
        stats = ideal.groupby("latent_dim")["accuracy_compressed"].mean().reset_index()
        plt.plot(stats["latent_dim"], stats["accuracy_compressed"], marker="o", label=f"{dataset} (canal ideal)")
        plotted = True
    if not plotted:
        plt.close()
        return
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


def plot_accuracy_vs_snr(df: pd.DataFrame, out_dir: str) -> None:
    """Acurácia vs. SNR do canal, no regime casado (SNR de treino = de teste)."""
    _ensure_dir(out_dir)
    compressed = df[df["accuracy_compressed"].notna()]
    if compressed.empty or "snr_test_db" not in compressed.columns:
        return
    # Regime casado: separa o efeito da SNR do efeito de mismatch treino/teste.
    matched = compressed[
        compressed["snr_train_db"].eq(compressed["snr_test_db"])
        | (compressed["snr_train_db"].isna() & compressed["snr_test_db"].isna())
    ]
    matched = matched[matched["snr_test_db"].notna()]
    if matched.empty:
        return

    plt.figure()
    for (dataset, latent_dim), group in matched.groupby(["dataset", "latent_dim"]):
        stats = (
            group.groupby("snr_test_db")["accuracy_compressed"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("snr_test_db")
        )
        line = plt.plot(
            stats["snr_test_db"], stats["mean"], marker="D",
            label=fr"{dataset} $L={int(latent_dim)}$",
        )[0]
        # Faixa de +/- 1 desvio-padrão só faz sentido com múltiplas seeds.
        if (stats["count"] > 1).any():
            plt.fill_between(
                stats["snr_test_db"],
                stats["mean"] - stats["std"].fillna(0),
                stats["mean"] + stats["std"].fillna(0),
                alpha=0.15, color=line.get_color(),
            )
    plt.xlabel("SNR do Canal (dB)")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.legend(ncol=1, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_snr.png"))
    plt.close()


def plot_snr_mismatch(df: pd.DataFrame, out_dir: str) -> None:
    """Matriz SNR de treino x SNR de teste: robustez a condições não vistas."""
    _ensure_dir(out_dir)
    compressed = df[df["accuracy_compressed"].notna()]
    if compressed.empty or "snr_train_db" not in compressed.columns:
        return
    grid = compressed.dropna(subset=["snr_train_db", "snr_test_db"])
    if grid["snr_train_db"].nunique() < 2 or grid["snr_test_db"].nunique() < 2:
        return

    for (dataset, latent_dim), group in grid.groupby(["dataset", "latent_dim"]):
        matrix = group.pivot_table(
            index="snr_train_db", columns="snr_test_db",
            values="accuracy_compressed", aggfunc="mean",
        )
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            continue
        plt.figure()
        im = plt.imshow(matrix.values, cmap="viridis", aspect="auto", origin="lower")
        plt.colorbar(im, label="Acurácia")
        plt.xticks(range(len(matrix.columns)), [f"{c:g}" for c in matrix.columns])
        plt.yticks(range(len(matrix.index)), [f"{i:g}" for i in matrix.index])
        plt.xlabel("SNR de Teste (dB)")
        plt.ylabel("SNR de Treino (dB)")
        plt.title(fr"{dataset}, $L={int(latent_dim)}$")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"snr_mismatch_{dataset}_L{int(latent_dim)}.png"))
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
                snr = row.get("snr_train_db")
                snr_txt = "ideal" if pd.isna(snr) else f"{snr:g} dB"
                label = fr"$L={int(row['latent_dim'])}$, SNR={snr_txt}"
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
    sort_cols = [c for c in ["dataset", "latent_dim", "snr_train_db", "snr_test_db"] if c in df.columns]
    df = df.sort_values(by=sort_cols)
    plot_accuracy_vs_compression(df, out_dir)
    plot_accuracy_vs_latent_dim(df, out_dir)
    plot_comm_cost_vs_latent_dim(df, out_dir)
    plot_accuracy_vs_snr(df, out_dir)
    plot_snr_mismatch(df, out_dir)
    plot_accuracy_vs_round(os.path.join(data_dir, "history.csv"), out_dir)


if __name__ == "__main__":
    generate_plots("./results/data", "./results/plots")
