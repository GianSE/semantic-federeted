import argparse
from typing import Dict

import torch
from torch import nn

from channel import parse_snr
from comm_cost import comm_summary
from data import client_label_counts, get_federated_dataloaders
from device import get_device, loader_kwargs
from federated import federated_train, set_seed
from metrics import accuracy_from_logits
from model_classifier import build_classifier
from save_results import save_run


def _train_step_fn(loss_fn: nn.Module):
    def step(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy_from_logits(logits, targets),
        }
        return loss, metrics

    return step


def _eval_step_fn(loss_fn: nn.Module):
    def step(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy_from_logits(logits, targets),
        }
        return loss, metrics

    return step


def run_baseline(config: Dict) -> Dict:
    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))

    client_loaders, test_loader = get_federated_dataloaders(
        dataset_name=config["dataset"],
        num_clients=config["num_clients"],
        batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
        train_fraction=config.get("train_fraction", 1.0),
        beta=config.get("beta"),
        **loader_kwargs(device, config.get("num_workers", 0)),
    )

    model = build_classifier(dataset_name=config["dataset"], input_type="raw")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=config["lr"])

    _, history = federated_train(
        global_model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        rounds=config["rounds"],
        local_epochs=config["local_epochs"],
        optimizer_fn=optimizer_fn,
        train_step_fn=_train_step_fn(loss_fn),
        eval_step_fn=_eval_step_fn(loss_fn),
        device=device,
        weight_snr_db=config.get("weight_snr_db"),
        show_progress=config.get("show_progress", True),
    )

    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    final_eval = history[-1]
    metrics = {
        "accuracy_baseline": final_eval["eval_accuracy"],
        "accuracy_compressed": None,
        "classification_loss": final_eval["eval_loss"],
        "reconstruction_loss": None,
        "total_samples": total_samples,
        **comm_summary(
            dataset_name=config["dataset"],
            num_samples=total_samples,
            model=model,
            num_clients=config["num_clients"],
            rounds=config["rounds"],
            latent_dim=None,  # baseline transmite a imagem bruta
        ),
    }
    return {
        "metrics": metrics,
        "history": history,
        "device": str(device),
        "client_distribution": client_label_counts(
            [loader.dataset for loader in client_loaders]
        ),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Federated baseline training")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-snr-db", type=parse_snr, default=None,
        help="SNR do uplink de pesos do FedAvg, em dB. 'none' = enlace ideal.",
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Concentracao de Dirichlet para particao non-IID. Omitir = IID.",
    )
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--runs-dir", type=str, default="./results/runs")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = vars(args)
    runs_dir = config.pop("runs_dir")
    config["model"] = "baseline"

    result = run_baseline(config)
    save_run(
        runs_dir, config, result["metrics"], result["history"], result["device"],
        extra={"client_distribution": result["client_distribution"]},
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()
