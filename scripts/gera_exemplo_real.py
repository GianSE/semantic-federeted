import argparse
import os

import matplotlib
matplotlib.use("Agg")  # script roda sem display (terminal/Colab headless)
import matplotlib.pyplot as plt
import torch

from semantic_federated.data import get_federated_dataloaders
from semantic_federated.models.autoencoder import build_autoencoder
from semantic_federated.models.classifier import LatentClassifier
from semantic_federated.reporting.checkpoints import checkpoint_path, load_checkpoint
from semantic_federated.training.compressed import CompressedModel


def _to_displayable(img_tensor: torch.Tensor):
    """Normalização min-max por imagem só para exibição (os pesos de treino usam a normalização do dataset)."""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def load_trained_model(config: dict, checkpoint_file: str = None, checkpoint_dir: str = "./results/checkpoints"):
    path = checkpoint_file or checkpoint_path(config, checkpoint_dir)
    record = load_checkpoint(path)

    autoencoder = build_autoencoder(config["dataset"], latent_dim=config["latent_dim"])
    classifier = LatentClassifier(latent_dim=config["latent_dim"])
    model = CompressedModel(autoencoder, classifier)
    model.load_state_dict(record["state_dict"])
    model.eval()
    return model, path


def generate_real_example(
    dataset: str = "cifar10",
    latent_dim: int = 64,
    noise_level: float = 0.0,
    channel_type: str = "awgn",
    partition: str = "iid",
    seed: int = 42,
    checkpoint_file: str = None,
    checkpoint_dir: str = "./results/checkpoints",
    num_examples: int = 4,
    out_dir: str = "./results/plots",
) -> str:
    config = {
        "dataset": dataset,
        "latent_dim": latent_dim,
        "noise_level": noise_level,
        "channel_type": channel_type,
        "partition": partition,
        "seed": seed,
    }
    print("Carregando checkpoint treinado...")
    model, used_path = load_trained_model(config, checkpoint_file, checkpoint_dir)
    print(f"  -> {used_path}")

    print(f"Carregando {num_examples} imagens reais de teste ({dataset})...")
    _, test_loader = get_federated_dataloaders(
        dataset_name=dataset,
        num_clients=1,
        batch_size=num_examples,
        test_batch_size=num_examples,
        seed=seed,
    )
    images, _ = next(iter(test_loader))
    images = images[:num_examples]

    with torch.no_grad():
        z = model.autoencoder.encode(images)
        recon = model.autoencoder.decode(z)

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(num_examples, 3, figsize=(10, 2.6 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, 3)

    for row in range(num_examples):
        ax_img, ax_latent, ax_recon = axes[row]

        ax_img.imshow(_to_displayable(images[row]))
        ax_img.set_title("Original" if row == 0 else "")
        ax_img.axis("off")

        z_np = z[row].numpy()
        ax_latent.bar(range(len(z_np)), z_np, color="#2a78d6")
        ax_latent.set_title(f"Latente (L={latent_dim})" if row == 0 else "")
        ax_latent.tick_params(labelsize=7)

        ax_recon.imshow(_to_displayable(recon[row]))
        ax_recon.set_title("Reconstrução" if row == 0 else "")
        ax_recon.axis("off")

    fig.suptitle(
        f"{dataset} · L={latent_dim} · ruído={noise_level} · {channel_type} · {partition} · seed={seed}",
        fontsize=10,
    )
    fig.tight_layout()

    noise_str = f"{noise_level:g}"
    out_path = os.path.join(
        out_dir, f"latent_examples_{dataset}_L{latent_dim}_noise{noise_str}_{channel_type}_{partition}_seed{seed}.png"
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figura salva em: {out_path}")
    return out_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Original vs. reconstrução a partir de um checkpoint treinado")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--channel-type", type=str, default="awgn")
    parser.add_argument("--partition", type=str, default="iid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="Caminho direto do checkpoint (ignora os demais filtros).")
    parser.add_argument("--checkpoint-dir", type=str, default="./results/checkpoints")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="./results/plots")
    return parser


def main():
    args = build_arg_parser().parse_args()
    generate_real_example(
        dataset=args.dataset,
        latent_dim=args.latent_dim,
        noise_level=args.noise_level,
        channel_type=args.channel_type,
        partition=args.partition,
        seed=args.seed,
        checkpoint_file=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        num_examples=args.num_examples,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
