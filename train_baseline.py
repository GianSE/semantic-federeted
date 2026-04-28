import argparse
from typing import Dict

import torch
from torch import nn

from compression import model_update_bits_per_round, total_raw_bits
from data import get_federated_dataloaders
from federated import federated_train, set_seed
from metrics import accuracy_from_logits
from model_classifier import build_classifier


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
    device = torch.device("cpu")

    client_loaders, test_loader = get_federated_dataloaders(
        dataset_name=config["dataset"],
        num_clients=config["num_clients"],
        batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
    )

    model = build_classifier(dataset_name=config["dataset"], input_type="raw")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    if config["baseline_comm_mode"] == "model":
        bits_per_client_round = model_update_bits_per_round(model)

        def comm_cost_fn(_client_id, _num_samples):
            return bits_per_client_round
    elif config["baseline_comm_mode"] == "raw":

        def comm_cost_fn(_client_id, _num_samples):
            return 0
    else:
        raise ValueError("baseline_comm_mode must be 'model' or 'raw'")

    optimizer_fn = lambda params: torch.optim.Adam(params, lr=config["lr"])

    _, history, _ = federated_train(
        global_model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        rounds=config["rounds"],
        local_epochs=config["local_epochs"],
        optimizer_fn=optimizer_fn,
        train_step_fn=_train_step_fn(loss_fn),
        eval_step_fn=_eval_step_fn(loss_fn),
        comm_cost_fn=comm_cost_fn,
        device=device,
        show_progress=True,
    )

    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    if config["baseline_comm_mode"] == "raw":
        total_comm_bits = total_raw_bits(config["dataset"], total_samples)
    else:
        total_comm_bits = bits_per_client_round * len(client_loaders)

    final_eval = history[-1]
    result = {
        "dataset": config["dataset"],
        "latent_dim": None,
        "noise_level": 0.0,
        "baseline_comm_mode": config["baseline_comm_mode"],
        "accuracy_baseline": final_eval["eval_accuracy"],
        "accuracy_compressed": None,
        "classification_loss": final_eval["eval_loss"],
        "reconstruction_loss": None,
        "compression_ratio": 1.0,
        "communication_cost_bits": total_comm_bits,
    }
    return result


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
    parser.add_argument("--baseline-comm-mode", type=str, choices=["model", "raw"], default="model")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = vars(args)
    result = run_baseline(config)
    print(result)


if __name__ == "__main__":
    main()
