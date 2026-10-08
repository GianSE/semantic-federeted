import os

import matplotlib
matplotlib.use("Agg")  # script headless: só salva PNG, nunca precisa de janela/GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta categórica validada (ordem fixa, CVD-safe) -- ver dataviz skill / references/palette.md.
PALETTE_ORDER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
MUTED = "#898781"
GRID_COLOR = "#e1e0d9"

# Cor fixa por entidade: a condição "padrão" (iid, awgn) é sempre azul; alternativas
# ganham as cores seguintes na mesma ordem em que aparecem na paleta. Mantido global
# para que a mesma categoria tenha sempre a mesma cor em qualquer figura.
COLOR_MAP = {
    "iid": "#2a78d6",
    "awgn": "#2a78d6",
    "dirichlet": "#eb6834",
    "rayleigh": "#1baf7a",
    "rician": "#eda100",
}

# Configurações para estilo IEEE/Acadêmico
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.edgecolor": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.fontsize": 9,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "figure.figsize": (5, 4),
    "savefig.dpi": 300,
})


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, out_dir: str, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename))
    plt.close(fig)


def plot_accuracy_vs_compression(summary: pd.DataFrame, out_dir: str) -> None:
    """Acurácia vs. razão de compressão, condição limpa (sem ruído, AWGN, IID)."""
    _ensure_dir(out_dir)
    df = summary[
        summary["accuracy_compressed_mean"].notna()
        & (summary["noise_level"] == 0.0)
        & (summary["channel_type"] == "awgn")
        & (summary["partition"] == "iid")
    ]
    if df.empty:
        return

    fig, ax = plt.subplots()
    for dataset, group in df.groupby("dataset"):
        group = group.sort_values("compression_ratio")
        ax.errorbar(
            group["compression_ratio"], group["accuracy_compressed_mean"],
            yerr=group["accuracy_compressed_std"].fillna(0.0),
            marker="s", linestyle="--", color=COLOR_MAP["awgn"], capsize=3,
            label=f"{dataset} (comprimido)",
        )
        baseline_rows = summary[(summary["dataset"] == dataset) & summary["accuracy_baseline_mean"].notna()]
        if not baseline_rows.empty:
            baseline_acc = baseline_rows["accuracy_baseline_mean"].iloc[0]
            ax.axhline(baseline_acc, color=MUTED, linestyle=":", linewidth=1.2, label=f"{dataset} (baseline sem compressão)")

    ax.set_xlabel(r"Razão de Compressão (CR)")
    ax.set_ylabel("Acurácia")
    ax.grid(True)
    ax.legend()
    _save(fig, out_dir, "accuracy_vs_compression_ratio.png")


def plot_accuracy_vs_latent_dim(summary: pd.DataFrame, out_dir: str) -> None:
    """Acurácia vs. dimensão latente, com ajuste logarítmico sobre os pontos médios."""
    _ensure_dir(out_dir)
    df = summary[
        summary["accuracy_compressed_mean"].notna()
        & (summary["noise_level"] == 0.0)
        & (summary["channel_type"] == "awgn")
        & (summary["partition"] == "iid")
    ]
    if df.empty:
        return

    fig, ax = plt.subplots()
    for dataset, group in df.groupby("dataset"):
        group = group.sort_values("latent_dim")
        x = group["latent_dim"].to_numpy(dtype=float)
        y = group["accuracy_compressed_mean"].to_numpy(dtype=float)
        yerr = group["accuracy_compressed_std"].fillna(0.0).to_numpy(dtype=float)

        ax.errorbar(
            x, y, yerr=yerr, marker="o", linestyle="none",
            color=COLOR_MAP["awgn"], capsize=3, label=f"{dataset} (observado)",
        )

        if len(x) >= 2:
            # Ajuste y = a*ln(x) + b sobre os pontos médios (sugestão de revisor: checar se a
            # queda de acurácia com a compressão é logarítmica em vez de linear).
            a, b = np.polyfit(np.log(x), y, 1)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = a * np.log(x_fit) + b
            y_pred = a * np.log(x) + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ax.plot(
                x_fit, y_fit, linestyle="--", color="#eb6834",
                label=rf"ajuste log ($R^2$={r2:.3f})",
            )

    ax.set_xlabel(r"Dimensão do Espaço Latente ($L$)")
    ax.set_ylabel("Acurácia")
    ax.grid(True)
    ax.legend()
    _save(fig, out_dir, "accuracy_vs_latent_dim.png")


def plot_reconstruction_loss_vs_latent_dim(summary: pd.DataFrame, out_dir: str) -> None:
    """Qualidade da reconstrução (MSE do Decoder) vs. dimensão latente, condição limpa.

    Complementa accuracy_vs_latent_dim: mostra o preço em fidelidade visual (não em
    acurácia da tarefa) de comprimir mais -- o Decoder nunca participa da inferência,
    mas sua perda aqui mede o quanto o espaço latente ainda carrega estrutura reconstruível.
    """
    _ensure_dir(out_dir)
    df = summary[
        summary["reconstruction_loss_mean"].notna()
        & (summary["noise_level"] == 0.0)
        & (summary["channel_type"] == "awgn")
        & (summary["partition"] == "iid")
    ]
    if df.empty:
        return

    fig, ax = plt.subplots()
    for dataset, group in df.groupby("dataset"):
        group = group.sort_values("latent_dim")
        ax.errorbar(
            group["latent_dim"], group["reconstruction_loss_mean"],
            yerr=group["reconstruction_loss_std"].fillna(0.0),
            marker="o", color=COLOR_MAP["awgn"], capsize=3, label=dataset,
        )
    ax.set_xlabel(r"Dimensão do Espaço Latente ($L$)")
    ax.set_ylabel("Erro de Reconstrução (MSE)")
    ax.grid(True)
    ax.legend()
    _save(fig, out_dir, "reconstruction_loss_vs_latent_dim.png")


def plot_comm_cost_vs_latent_dim(summary: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    df = summary[
        summary["accuracy_compressed_mean"].notna()
        & (summary["noise_level"] == 0.0)
        & (summary["channel_type"] == "awgn")
        & (summary["partition"] == "iid")
    ]
    if df.empty:
        return

    fig, ax = plt.subplots()
    for dataset, group in df.groupby("dataset"):
        group = group.drop_duplicates("latent_dim").sort_values("latent_dim")
        ax.semilogy(
            group["latent_dim"], group["communication_cost_bits"],
            marker="^", color=COLOR_MAP["awgn"], label=dataset,
        )
    ax.set_xlabel(r"Dimensão do Espaço Latente ($L$)")
    ax.set_ylabel("Custo de Comunicação (bits)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, out_dir, "communication_cost_vs_latent_dim.png")


def plot_accuracy_vs_noise(summary: pd.DataFrame, out_dir: str) -> None:
    """Acurácia vs. nível de ruído, por dimensão latente (até 4 séries, condição IID/AWGN)."""
    _ensure_dir(out_dir)
    df = summary[
        summary["accuracy_compressed_mean"].notna()
        & (summary["channel_type"] == "awgn")
        & (summary["partition"] == "iid")
    ]
    if df.empty:
        return

    fig, ax = plt.subplots()
    combos = sorted(df.groupby(["dataset", "latent_dim"]).groups.keys(), key=lambda c: c[1])
    for i, (dataset, latent_dim) in enumerate(combos[: len(PALETTE_ORDER)]):
        group = df[(df["dataset"] == dataset) & (df["latent_dim"] == latent_dim)].sort_values("noise_level")
        ax.errorbar(
            group["noise_level"], group["accuracy_compressed_mean"],
            yerr=group["accuracy_compressed_std"].fillna(0.0),
            marker="D", color=PALETTE_ORDER[i], capsize=3,
            label=fr"{dataset} $L={int(latent_dim)}$",
        )
    ax.set_xlabel(r"Nível de Ruído ($\sigma$)")
    ax.set_ylabel("Acurácia")
    ax.grid(True)
    ax.legend(loc="best")
    _save(fig, out_dir, "accuracy_vs_noise_level.png")


def _grouped_bar(ax, categories, series: dict, colors: dict) -> None:
    """series: {nome_da_serie: (valores_y, valores_yerr)}, alinhados com `categories`."""
    num_series = len(series)
    width = 0.8 / num_series
    x = np.arange(len(categories))
    for i, (name, (values, errors)) in enumerate(series.items()):
        offset = (i - (num_series - 1) / 2) * width
        ax.bar(
            x + offset, values, width=width * 0.9, yerr=errors,
            color=colors[name], label=name, capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories])


def plot_partition_comparison(summary: pd.DataFrame, out_dir: str) -> None:
    """Compara acurácia sob particionamento IID vs. não-IID (Dirichlet), mesma config de canal."""
    _ensure_dir(out_dir)
    df = summary[summary["accuracy_compressed_mean"].notna() & (summary["channel_type"] == "awgn")]
    partitions = df["partition"].unique().tolist()
    if "iid" not in partitions or "dirichlet" not in partitions:
        return  # sem dados das duas partições ainda -- nada a comparar

    # Usa a dimensão latente com dados em ambas as partições.
    shared_latent_dims = set(df[df["partition"] == "iid"]["latent_dim"]) & set(
        df[df["partition"] == "dirichlet"]["latent_dim"]
    )
    if not shared_latent_dims:
        return
    latent_dim = sorted(shared_latent_dims)[0]
    subset = df[df["latent_dim"] == latent_dim]

    noise_levels = sorted(
        set(subset[subset["partition"] == "iid"]["noise_level"])
        & set(subset[subset["partition"] == "dirichlet"]["noise_level"])
    )
    if not noise_levels:
        return

    series = {}
    for partition in ["iid", "dirichlet"]:
        rows = subset[subset["partition"] == partition].set_index("noise_level").loc[noise_levels]
        series[partition] = (rows["accuracy_compressed_mean"].to_numpy(), rows["accuracy_compressed_std"].fillna(0.0).to_numpy())

    fig, ax = plt.subplots()
    _grouped_bar(ax, noise_levels, series, COLOR_MAP)
    ax.set_xlabel(r"Nível de Ruído ($\sigma$)")
    ax.set_ylabel("Acurácia")
    ax.set_title(rf"IID vs. não-IID ($L={int(latent_dim)}$)")
    ax.grid(True, axis="y")
    ax.legend()
    _save(fig, out_dir, "accuracy_iid_vs_dirichlet.png")


def plot_channel_comparison(summary: pd.DataFrame, out_dir: str) -> None:
    """Compara acurácia sob AWGN vs. canais com desvanecimento (Rayleigh/Rician), mesma partição."""
    _ensure_dir(out_dir)
    df = summary[summary["accuracy_compressed_mean"].notna() & (summary["partition"] == "iid")]
    channel_types = [c for c in df["channel_type"].unique().tolist() if pd.notna(c)]
    if len(channel_types) < 2:
        return  # só rodou AWGN ainda -- nada a comparar

    shared_latent_dims = None
    for channel in channel_types:
        dims = set(df[df["channel_type"] == channel]["latent_dim"])
        shared_latent_dims = dims if shared_latent_dims is None else (shared_latent_dims & dims)
    if not shared_latent_dims:
        return
    latent_dim = sorted(shared_latent_dims)[0]
    subset = df[df["latent_dim"] == latent_dim]

    noise_levels = None
    for channel in channel_types:
        levels = set(subset[subset["channel_type"] == channel]["noise_level"])
        noise_levels = levels if noise_levels is None else (noise_levels & levels)
    noise_levels = sorted(noise_levels) if noise_levels else []
    if not noise_levels:
        return

    channel_order = [c for c in ["awgn", "rayleigh", "rician"] if c in channel_types]
    series = {}
    for channel in channel_order:
        rows = subset[subset["channel_type"] == channel].set_index("noise_level").loc[noise_levels]
        series[channel] = (rows["accuracy_compressed_mean"].to_numpy(), rows["accuracy_compressed_std"].fillna(0.0).to_numpy())

    fig, ax = plt.subplots()
    _grouped_bar(ax, noise_levels, series, COLOR_MAP)
    ax.set_xlabel(r"Nível de Ruído ($\sigma$)")
    ax.set_ylabel("Acurácia")
    ax.set_title(rf"Modelos de canal ($L={int(latent_dim)}$, IID)")
    ax.grid(True, axis="y")
    ax.legend()
    _save(fig, out_dir, "accuracy_by_channel_type.png")


def generate_plots(summary_csv: str, out_dir: str) -> None:
    summary = pd.read_csv(summary_csv)
    summary = summary.sort_values(by=["dataset", "latent_dim", "noise_level"])
    plot_accuracy_vs_compression(summary, out_dir)
    plot_accuracy_vs_latent_dim(summary, out_dir)
    plot_reconstruction_loss_vs_latent_dim(summary, out_dir)
    plot_comm_cost_vs_latent_dim(summary, out_dir)
    plot_accuracy_vs_noise(summary, out_dir)
    plot_partition_comparison(summary, out_dir)
    plot_channel_comparison(summary, out_dir)


if __name__ == "__main__":
    generate_plots("./results/tables/results_summary.csv", "./results/plots")
