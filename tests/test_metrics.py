import pytest
import torch

from semantic_federated.metrics import accuracy_from_logits, average_metrics


def test_accuracy_from_logits_all_correct():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    assert accuracy_from_logits(logits, targets) == 1.0


def test_accuracy_from_logits_half_correct():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 1])  # segunda amostra está errada
    assert accuracy_from_logits(logits, targets) == 0.5


def test_accuracy_from_logits_empty_batch_does_not_divide_by_zero():
    logits = torch.empty((0, 2))
    targets = torch.empty((0,), dtype=torch.long)
    assert accuracy_from_logits(logits, targets) == 0.0


def test_average_metrics_computes_mean_per_key():
    metrics_list = [
        {"loss": 1.0, "accuracy": 0.5},
        {"loss": 3.0, "accuracy": 0.7},
    ]
    result = average_metrics(metrics_list)
    assert result["loss"] == pytest.approx(2.0)
    assert result["accuracy"] == pytest.approx(0.6)


def test_average_metrics_empty_list_returns_empty_dict():
    assert average_metrics([]) == {}
