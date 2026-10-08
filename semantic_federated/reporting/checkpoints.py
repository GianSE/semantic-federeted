import os
from typing import Dict

import torch
from torch import nn

DEFAULT_CHECKPOINT_DIR = "./results/checkpoints"


def checkpoint_filename(config: Dict) -> str:
    """Nome determinístico a partir da configuração, para localizar o checkpoint certo depois."""
    noise = f"{config['noise_level']:g}"
    channel = config.get("channel_type", "awgn")
    partition = config.get("partition", "iid")
    return (
        f"{config['dataset']}_L{config['latent_dim']}_noise{noise}"
        f"_{channel}_{partition}_seed{config['seed']}.pt"
    )


def checkpoint_path(config: Dict, out_dir: str = DEFAULT_CHECKPOINT_DIR) -> str:
    return os.path.join(out_dir, checkpoint_filename(config))


def save_checkpoint(model: nn.Module, config: Dict, out_dir: str = DEFAULT_CHECKPOINT_DIR) -> str:
    """Salva os pesos treinados (Encoder + Classificador + Decoder) junto com a config que os gerou."""
    os.makedirs(out_dir, exist_ok=True)
    path = checkpoint_path(config, out_dir)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    return path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    if not os.path.isfile(path):
        available = []
        out_dir = os.path.dirname(path) or DEFAULT_CHECKPOINT_DIR
        if os.path.isdir(out_dir):
            available = sorted(os.listdir(out_dir))
        raise FileNotFoundError(
            f"Checkpoint nao encontrado: {path}\n"
            f"Disponiveis em {out_dir}: {available or '(nenhum)'}"
        )
    return torch.load(path, map_location=map_location, weights_only=False)
