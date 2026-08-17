"""Orquestrador da grade de experimentos.

A grade e *retomavel*: cada configuracao vira um arquivo em `results/runs/` e
execucoes subsequentes pulam o que ja existe. Isso permite acumular resultados
em varias sessoes (util em CPU) e migrar para GPU sem refazer o que ja rodou.

Perfis de uso:
  smoke  python main.py --datasets mnist --latent-dims 16 --noise-levels 0.0 \
                        --num-clients 2 --rounds 2 --train-fraction 0.05
  dev    python main.py --datasets cifar10 --latent-dims 16 64 --noise-levels 0.0 0.05 \
                        --num-clients 5 --rounds 5
  paper  ver README (grade completa, recomendada em GPU)
"""

import argparse
from typing import Dict, List

from plot_results import generate_plots
from save_results import export_results, run_exists, run_id, save_run
from tables import generate_tables
from train_baseline import run_baseline
from train_compressed import run_compressed


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run full FL compression experiments")
    parser.add_argument("--datasets", type=str, nargs="+", default=["mnist", "cifar10"])
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--runs-dir", type=str, default="./results/runs")
    parser.add_argument("--out-dir", type=str, default="./results")
    parser.add_argument("--force", action="store_true", help="Refaz runs ja existentes")
    parser.add_argument("--export-only", action="store_true", help="So reconstroi CSVs/figuras")
    return parser


def _shared(args, dataset: str, seed: int) -> Dict:
    return {
        "dataset": dataset,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "lr": args.lr,
        "seed": seed,
        "train_fraction": args.train_fraction,
    }


def build_grid(args) -> List[Dict]:
    """Enumera as configuracoes da grade, sem executar nada."""
    configs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            base = _shared(args, dataset, seed)
            configs.append({**base, "model": "baseline"})
            for latent_dim in args.latent_dims:
                for noise_level in args.noise_levels:
                    configs.append(
                        {
                            **base,
                            "model": "compressed",
                            "latent_dim": latent_dim,
                            "noise_level": noise_level,
                            "dropout_p": args.dropout_p,
                            "alpha": args.alpha,
                        }
                    )
    return configs


def main():
    args = build_arg_parser().parse_args()

    if not args.export_only:
        configs = build_grid(args)
        pending = [c for c in configs if args.force or not run_exists(args.runs_dir, c)]
        print(f"Grade: {len(configs)} config(s) | pendentes: {len(pending)} | ja em disco: {len(configs) - len(pending)}")

        for index, config in enumerate(pending, start=1):
            label = config.get("model")
            if label == "compressed":
                label += f" L={config['latent_dim']} sigma={config['noise_level']}"
            print(f"\n[{index}/{len(pending)}] {config['dataset']} seed={config['seed']} {label} ({run_id(config)})")

            runner = run_baseline if config["model"] == "baseline" else run_compressed
            result = runner({**config, "device": args.device, "num_workers": args.num_workers})
            save_run(args.runs_dir, config, result["metrics"], result["history"], result["device"])

    data_dir = f"{args.out_dir}/data"
    if export_results(args.runs_dir, data_dir) is not None:
        generate_plots(data_dir, f"{args.out_dir}/plots")
        generate_tables(f"{data_dir}/experiment_results.csv", f"{args.out_dir}/tables")


if __name__ == "__main__":
    main()
