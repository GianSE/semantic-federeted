from typing import Dict, List

import torch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    totals = {key: 0.0 for key in keys}
    for metrics in metrics_list:
        for key in keys:
            totals[key] += metrics[key]
    return {key: totals[key] / len(metrics_list) for key in keys}
