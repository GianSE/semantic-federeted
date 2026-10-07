import torch

from semantic_federated.federated import average_state_dicts


def test_average_state_dicts_equal_weights_is_plain_mean():
    state_dicts = [
        {"w": torch.tensor([0.0, 0.0])},
        {"w": torch.tensor([2.0, 4.0])},
    ]
    avg = average_state_dicts(state_dicts, weights=[1, 1])
    assert torch.allclose(avg["w"], torch.tensor([1.0, 2.0]))


def test_average_state_dicts_weights_by_num_samples():
    # Cliente A tem 1 amostra (peso 1), cliente B tem 3 amostras (peso 3):
    # a media deve pesar 3x mais para o cliente B (Eq. 3 do paper: w_{t+1} = sum(n_k/n * w_t^k)).
    state_dicts = [
        {"w": torch.tensor([0.0])},
        {"w": torch.tensor([4.0])},
    ]
    avg = average_state_dicts(state_dicts, weights=[1, 3])
    assert torch.allclose(avg["w"], torch.tensor([3.0]))


def test_average_state_dicts_empty_input_returns_empty_dict():
    assert average_state_dicts([], weights=[]) == {}
