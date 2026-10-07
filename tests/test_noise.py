import torch

from semantic_federated.noise import add_gaussian_noise, apply_dropout_noise


def test_add_gaussian_noise_sigma_zero_is_identity():
    z = torch.ones(8)
    assert torch.equal(add_gaussian_noise(z, sigma=0.0), z)


def test_add_gaussian_noise_changes_values_when_sigma_positive():
    torch.manual_seed(0)
    z = torch.zeros(1000)
    noisy = add_gaussian_noise(z, sigma=0.1)
    assert not torch.equal(noisy, z)
    # Com 1000 amostras e sigma=0.1, o desvio padrão amostral deve ficar perto de 0.1.
    assert abs(noisy.std().item() - 0.1) < 0.02


def test_apply_dropout_noise_p_zero_is_identity():
    z = torch.ones(8)
    assert torch.equal(apply_dropout_noise(z, dropout_p=0.0, training=True), z)


def test_apply_dropout_noise_eval_mode_is_identity_even_with_p_positive():
    z = torch.ones(8)
    assert torch.equal(apply_dropout_noise(z, dropout_p=0.5, training=False), z)
