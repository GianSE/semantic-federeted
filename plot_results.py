import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_accuracy_vs_compression(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    compressed = df[df["accuracy_compressed"].notna()]
    if compressed.empty:
        return
    plt.figure()
    for dataset, group in compressed.groupby("dataset"):
        plt.plot(group["compression_ratio"], group["accuracy_compressed"], marker="o", label=dataset)
    plt.xlabel("Compression Ratio")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Compression Ratio")
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
        plt.plot(group["latent_dim"], group["accuracy_compressed"], marker="o", label=dataset)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latent Dimension")
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
        plt.plot(group["latent_dim"], group["communication_cost_bits"], marker="o", label=dataset)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Communication Cost (bits)")
    plt.title("Communication Cost vs Latent Dimension")
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
    for (dataset, latent_dim), group in compressed.groupby(["dataset", "latent_dim"]):
        plt.plot(group["noise_level"], group["accuracy_compressed"], marker="o", label=f"{dataset}-z{latent_dim}")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Noise Level")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_noise_level.png"))
    plt.close()


def generate_plots(results_csv: str, out_dir: str) -> None:
    df = pd.read_csv(results_csv)
    plot_accuracy_vs_compression(df, out_dir)
    plot_accuracy_vs_latent_dim(df, out_dir)
    plot_comm_cost_vs_latent_dim(df, out_dir)
    plot_accuracy_vs_noise(df, out_dir)


if __name__ == "__main__":
    generate_plots("./results/data/experiment_results.csv", "./results/plots")
