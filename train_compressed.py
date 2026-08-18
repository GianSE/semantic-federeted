import argparse
from typing import Dict, Optional

import torch
from torch import nn

from comm_cost import comm_summary
from data import client_label_counts, get_federated_dataloaders
from device import get_device, loader_kwargs
from federated import federated_train, set_seed
from metrics import accuracy_from_logits
from model_autoencoder import build_autoencoder
from model_classifier import LatentClassifier
from channel import (
    apply_channel,
    apply_dropout_noise,
    normalize_power,
    parse_snr,
    quantize_latent,
    resolve_test_snr,
)
from save_results import save_run


class CompressedModel(nn.Module):
    """Transmissor (encoder) -> canal -> receptor (classificador + decoder).

    O latente e normalizado em potencia antes do canal, de modo que a condicao
    de canal seja definida por uma SNR em dB e nao por uma amplitude absoluta
    que o proprio treinamento poderia anular aumentando ||z||.
    """

    def __init__(self, autoencoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier

    def forward(
        self,
        x,
        snr_db,
        dropout_p: float,
        training: bool,
        channel: str = "awgn",
        latent_bits: Optional[int] = None,
        rician_k_db: Optional[float] = None,
    ):
        # Transmissor: codifica, normaliza a potencia e quantiza para o formato
        # efetivamente transmitido (e contabilizado em bits).
        z = normalize_power(self.autoencoder.encode(x))
        z = quantize_latent(z, latent_bits)
        z_hat = apply_channel(z, snr_db, channel=channel, rician_k_db=rician_k_db)
        z_hat = apply_dropout_noise(z_hat, dropout_p, training=training)
        # Classificador e decoder estao no receptor: ambos operam sobre o
        # latente efetivamente recebido (z_hat), conforme a Eq. (2) do artigo.
        logits = self.classifier(z_hat)
        recon = self.autoencoder.decode(z_hat)
        return z, z_hat, logits, recon


def _step_fn(
    loss_fn: nn.Module,
    recon_loss_fn: nn.Module,
    alpha: float,
    snr_db,
    dropout_p: float,
    channel: str,
    latent_bits: Optional[int],
    rician_k_db: Optional[float],
    training: bool,
):
    def step(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, _, logits, recon = model(
            inputs, snr_db, dropout_p, training=training,
            channel=channel, latent_bits=latent_bits, rician_k_db=rician_k_db,
        )
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
        beta=config.get("beta"),
        **loader_kwargs(device, config.get("num_workers", 0)),
    )

    autoencoder = build_autoencoder(config["dataset"], latent_dim=config["latent_dim"])
    classifier = LatentClassifier(latent_dim=config["latent_dim"])
    model = CompressedModel(autoencoder, classifier)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=config["lr"])

    channel = config.get("channel", "awgn")
    latent_bits = config.get("latent_bits")
    rician_k_db = config.get("rician_k_db")
    common = (loss_fn, recon_loss_fn, config["alpha"])
    tail = (config["dropout_p"], channel, latent_bits, rician_k_db)
    train_args = (*common, config["snr_train_db"], *tail)
    eval_args = (*common, config["snr_test_db"], *tail)

    _, history = federated_train(
        global_model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        rounds=config["rounds"],
        local_epochs=config["local_epochs"],
        optimizer_fn=optimizer_fn,
        train_step_fn=_step_fn(*train_args, training=True),
        eval_step_fn=_step_fn(*eval_args, training=False),
        device=device,
        weight_snr_db=config.get("weight_snr_db"),
        show_progress=config.get("show_progress", True),
    )

    final_eval = history[-1]
    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    metrics = {
        "accuracy_baseline": None,
        "accuracy_compressed": final_eval["eval_accuracy"],
        "classification_loss": final_eval["eval_classification_loss"],
        "reconstruction_loss": final_eval["eval_reconstruction_loss"],
        "total_samples": total_samples,
        **comm_summary(
            dataset_name=config["dataset"],
            num_samples=total_samples,
            model=model,
            num_clients=config["num_clients"],
            rounds=config["rounds"],
            latent_dim=config["latent_dim"],
            latent_bits=latent_bits if latent_bits is not None else 32,
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
    parser = argparse.ArgumentParser(description="Federated compressed training")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--snr-train-db",
        type=parse_snr,
        default=None,
        help="SNR do canal no treino, em dB. Use 'none' para canal ideal.",
    )
    parser.add_argument(
        "--snr-test-db",
        type=parse_snr,
        default="match",
        help="SNR na avaliacao. 'match' usa a mesma do treino.",
    )
    parser.add_argument(
        "--channel", type=str, default="awgn",
        choices=["awgn", "rayleigh", "rician"],
    )
    parser.add_argument(
        "--rician-k-db", type=float, default=None,
        help="Fator K de Rice em dB (apenas para --channel rician).",
    )
    parser.add_argument(
        "--latent-bits",
        type=int,
        default=32,
        help="Bits por dimensao latente transmitida (32 = sem quantizacao).",
    )
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
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
    config["model"] = "compressed"
    config["snr_test_db"] = resolve_test_snr(config["snr_test_db"], config["snr_train_db"])

    result = run_compressed(config)
    save_run(
        runs_dir, config, result["metrics"], result["history"], result["device"],
        extra={"client_distribution": result["client_distribution"]},
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()
