"""Seleção de dispositivo (CPU/CUDA) e opções de DataLoader associadas.

O default é "auto": usa CUDA quando disponível e CPU caso contrário. Assim o
mesmo comando roda sem alteração numa máquina local sem GPU e num ambiente com
GPU (Colab/Kaggle).

Observação sobre reprodutibilidade: resultados não são bit-idênticos entre CPU e
CUDA. O dispositivo efetivamente usado é registrado em cada run (ver
save_results.py) para permitir auditoria posterior.
"""

import torch


def get_device(spec: str = "auto") -> torch.device:
    spec = spec.lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda foi solicitado, mas torch.cuda.is_available() e False. "
            "Use --device cpu ou --device auto."
        )
    return torch.device(spec)


def loader_kwargs(device: torch.device, num_workers: int = 0) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs
