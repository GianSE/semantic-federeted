import argparse
import os
from typing import Dict, List

from compression import latent_bits_per_sample
from data import get_federated_dataloaders
from plot_results import generate_plots
from save_results import save_results
from tables import generate_tables
from train_baseline import run_baseline
from train_compressed import run_compressed


def estimate_compressed_total_bits(dataset: str, latent_dim: int, rounds: int, num_clients: int, seed: int) -> int:
    client_loaders, _ = get_federated_dataloaders(
        dataset_name=dataset,
        num_clients=num_clients,
        batch_size=32,
        test_batch_size=256,
        seed=seed,
    )
    bits_per_sample = latent_bits_per_sample(latent_dim)
    total_bits = 0
    for loader in client_loaders:
        total_bits += bits_per_sample * len(loader.dataset) * rounds
    return total_bits


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run full FL compression experiments")
    parser.add_argument("--datasets", type=str, nargs="+", default=["mnist", "cifar10"])
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-comm-mode", type=str, choices=["model", "raw"], default="raw")
    parser.add_argument("--fixed-comm-budget", type=int, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()

    results: List[Dict] = []

    for dataset in args.datasets:
        baseline_config = {
            "dataset": dataset,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "baseline_comm_mode": "raw",
        }
        results.append(run_baseline(baseline_config))

        for latent_dim in args.latent_dims:
            if args.fixed_comm_budget is not None:
                estimated_bits = estimate_compressed_total_bits(
                    dataset=dataset,
                    latent_dim=latent_dim,
                    rounds=args.rounds,
                    num_clients=args.num_clients,
                    seed=args.seed,
                )
                if estimated_bits > args.fixed_comm_budget:
                    continue
            for noise_level in args.noise_levels:
                compressed_config = {
                    "dataset": dataset,
                    "latent_dim": latent_dim,
                    "noise_level": noise_level,
                    "dropout_p": 0.0,
                    "num_clients": args.num_clients,
                    "rounds": args.rounds,
                    "local_epochs": args.local_epochs,
                    "batch_size": args.batch_size,
                    "test_batch_size": args.test_batch_size,
                    "lr": args.lr,
                    "alpha": args.alpha,
                    "seed": args.seed,
                }
                results.append(run_compressed(compressed_config))

    out_dir = "./results/data"
    save_results(results, out_dir, "experiment_results")

    generate_plots("./results/data/experiment_results.csv", "./results/plots")
    generate_tables("./results/data/experiment_results.csv", "./results/tables")


if __name__ == "__main__":
    main()
