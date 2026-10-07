import math

import torch

from semantic_federated.noise import (
    add_gaussian_noise,
    apply_channel,
    apply_dropout_noise,
    apply_rayleigh_fading,
    apply_rician_fading,
)


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


def test_apply_rayleigh_fading_scale_zero_is_identity():
    z = torch.ones(10, 4)
    assert torch.equal(apply_rayleigh_fading(z, scale=0.0, noise_sigma=0.0), z)


def test_apply_rayleigh_fading_mean_attenuation_matches_theory():
    torch.manual_seed(0)
    batch, dim = 20000, 1
    z = torch.ones(batch, dim)
    faded = apply_rayleigh_fading(z, scale=1.0, noise_sigma=0.0)
    # h e normalizado para ganho medio de potencia unitario (E[|h|^2]=1), o que da
    # E[|h|] = sqrt(pi)/2 ~= 0.8862 (nao sqrt(pi/2), que valeria sem a normalizacao).
    assert abs(faded.mean().item() - math.sqrt(math.pi) / 2) < 0.05


def test_apply_rician_fading_high_k_factor_is_almost_deterministic():
    torch.manual_seed(0)
    batch, dim = 20000, 1
    z = torch.ones(batch, dim)
    # K alto -> canal dominado pela linha de visada, pouca variabilidade em torno de scale.
    faded = apply_rician_fading(z, k_factor=1000.0, scale=2.0, noise_sigma=0.0)
    assert abs(faded.mean().item() - 2.0) < 0.1
    assert faded.std().item() < 0.1


def test_apply_channel_awgn_matches_add_gaussian_noise():
    torch.manual_seed(0)
    z = torch.zeros(100, 4)
    expected = add_gaussian_noise(z.clone(), sigma=0.1)
    torch.manual_seed(0)
    actual = apply_channel(z.clone(), channel_type="awgn", noise_sigma=0.1)
    assert torch.allclose(expected, actual)


def test_apply_channel_rejects_unknown_type():
    z = torch.ones(4, 2)
    try:
        apply_channel(z, channel_type="nope")
        assert False, "esperava ValueError para channel_type desconhecido"
    except ValueError:
        pass
