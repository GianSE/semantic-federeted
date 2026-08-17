from typing import Dict, List, Optional

import torch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def average_metrics(
    metrics_list: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Media das metricas, opcionalmente ponderada pelo tamanho de cada batch.

    Sem ponderacao, o ultimo batch (tipicamente menor) teria o mesmo peso dos
    demais, enviesando a acuracia reportada.
    """
    if not metrics_list:
        return {}
    if weights is None:
        weights = [1.0] * len(metrics_list)
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return {}
    keys = metrics_list[0].keys()
    return {
        key: sum(m[key] * w for m, w in zip(metrics_list, weights)) / total_weight
        for key in keys
    }
