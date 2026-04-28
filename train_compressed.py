import argparse
from typing import Dict

import torch
from torch import nn

from compression import compression_ratio, total_latent_bits, total_raw_bits
from data import get_federated_dataloaders
from federated import federated_train, set_seed
from metrics import accuracy_from_logits
from model_autoencoder import build_autoencoder
from model_classifier import LatentClassifier
from noise import add_gaussian_noise, apply_dropout_noise


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


def _train_step_fn(loss_fn: nn.Module, recon_loss_fn: nn.Module, alpha: float, noise_sigma: float, dropout_p: float):
    def step(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, _, logits, recon = model(inputs, noise_sigma, dropout_p, training=True)
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


def _eval_step_fn(loss_fn: nn.Module, recon_loss_fn: nn.Module, alpha: float, noise_sigma: float, dropout_p: float):
    def step(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, _, logits, recon = model(inputs, noise_sigma, dropout_p, training=False)
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
    device = torch.device("cpu")

    client_loaders, test_loader = get_federated_dataloaders(
        dataset_name=config["dataset"],
        num_clients=config["num_clients"],
        batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
    )

    autoencoder = build_autoencoder(config["dataset"], latent_dim=config["latent_dim"])
    classifier = LatentClassifier(latent_dim=config["latent_dim"])
    model = CompressedModel(autoencoder, classifier)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()

    def comm_cost_fn(_client_id, _num_samples):
        return 0

    optimizer_fn = lambda params: torch.optim.Adam(params, lr=config["lr"])

    _, history, _ = federated_train(
        global_model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        rounds=config["rounds"],
        local_epochs=config["local_epochs"],
        optimizer_fn=optimizer_fn,
        train_step_fn=_train_step_fn(
            loss_fn,
            recon_loss_fn,
            config["alpha"],
            config["noise_level"],
            config["dropout_p"],
        ),
        eval_step_fn=_eval_step_fn(
            loss_fn,
            recon_loss_fn,
            config["alpha"],
            config["noise_level"],
            config["dropout_p"],
        ),
        comm_cost_fn=comm_cost_fn,
        device=device,
        show_progress=True,
    )

    final_eval = history[-1]
    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    raw_bits = total_raw_bits(config["dataset"], total_samples)
    compressed_bits = total_latent_bits(config["latent_dim"], total_samples)
    total_comm_bits = compressed_bits
    result = {
        "dataset": config["dataset"],
        "latent_dim": config["latent_dim"],
        "noise_level": config["noise_level"],
        "baseline_comm_mode": None,
        "accuracy_baseline": None,
        "accuracy_compressed": final_eval["eval_accuracy"],
        "classification_loss": final_eval["eval_classification_loss"],
        "reconstruction_loss": final_eval["eval_reconstruction_loss"],
        "compression_ratio": compression_ratio(raw_bits, compressed_bits),
        "communication_cost_bits": total_comm_bits,
    }
    return result


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
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = vars(args)
    result = run_compressed(config)
    print(result)


if __name__ == "__main__":
    main()
