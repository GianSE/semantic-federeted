import argparse
from typing import Dict

import torch
from torch import nn

from comm_cost import compression_ratio, total_latent_bits, total_raw_bits
from data import get_federated_dataloaders
from device import get_device, loader_kwargs
from federated import federated_train, set_seed
from metrics import accuracy_from_logits
from model_autoencoder import build_autoencoder
from model_classifier import LatentClassifier
from noise import add_gaussian_noise, apply_dropout_noise
from save_results import save_run


class CompressedModel(nn.Module):
    def __init__(self, autoencoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier

    def forward(self, x, noise_sigma: float, dropout_p: float, training: bool):
        z = self.autoencoder.encode(x)
        z_noisy = add_gaussian_noise(z, noise_sigma)
        z_noisy = apply_dropout_noise(z_noisy, dropout_p, training=training)
        logits = self.classifier(z_noisy)
        recon = self.autoencoder.decode(z)
        return z, z_noisy, logits, recon


def _step_fn(
    loss_fn: nn.Module,
    recon_loss_fn: nn.Module,
    alpha: float,
    noise_sigma: float,
    dropout_p: float,
    training: bool,
):
    def step(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, _, logits, recon = model(inputs, noise_sigma, dropout_p, training=training)
        classification_loss = loss_fn(logits, targets)
        reconstruction_loss = recon_loss_fn(recon, inputs)
        loss = classification_loss + alpha * reconstruction_loss
        metrics = {
            "loss": loss.item(),
            "classification_loss": classification_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "accuracy": accuracy_from_logits(logits, targets),
        }
        return loss, metrics

    return step


def run_compressed(config: Dict) -> Dict:
    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))

    client_loaders, test_loader = get_federated_dataloaders(
        dataset_name=config["dataset"],
        num_clients=config["num_clients"],
        batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
        train_fraction=config.get("train_fraction", 1.0),
        **loader_kwargs(device, config.get("num_workers", 0)),
    )

    autoencoder = build_autoencoder(config["dataset"], latent_dim=config["latent_dim"])
    classifier = LatentClassifier(latent_dim=config["latent_dim"])
    model = CompressedModel(autoencoder, classifier)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=config["lr"])

    step_args = (loss_fn, recon_loss_fn, config["alpha"], config["noise_level"], config["dropout_p"])

    _, history = federated_train(
        global_model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        rounds=config["rounds"],
        local_epochs=config["local_epochs"],
        optimizer_fn=optimizer_fn,
        train_step_fn=_step_fn(*step_args, training=True),
        eval_step_fn=_step_fn(*step_args, training=False),
        device=device,
        show_progress=config.get("show_progress", True),
    )

    final_eval = history[-1]
    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    raw_bits = total_raw_bits(config["dataset"], total_samples)
    compressed_bits = total_latent_bits(config["latent_dim"], total_samples)
    metrics = {
        "accuracy_baseline": None,
        "accuracy_compressed": final_eval["eval_accuracy"],
        "classification_loss": final_eval["eval_classification_loss"],
        "reconstruction_loss": final_eval["eval_reconstruction_loss"],
        "compression_ratio": compression_ratio(raw_bits, compressed_bits),
        "communication_cost_bits": compressed_bits,
        "total_samples": total_samples,
    }
    return {"metrics": metrics, "history": history, "device": str(device)}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Federated compressed training")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--runs-dir", type=str, default="./results/runs")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = vars(args)
    runs_dir = config.pop("runs_dir")
    config["model"] = "compressed"

    result = run_compressed(config)
    save_run(runs_dir, config, result["metrics"], result["history"], result["device"])
    print(result["metrics"])


if __name__ == "__main__":
    main()
