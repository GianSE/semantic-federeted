import torch
from torch import nn

from semantic_federated.compression import (
    bits_for_shape,
    compression_ratio,
    input_dim_values,
    latent_bits_per_sample,
    model_update_bits,
    model_update_bits_per_round,
    raw_input_bits_per_sample,
    total_latent_bits,
    total_raw_bits,
)


def test_bits_for_shape():
    assert bits_for_shape((3, 32, 32), bits_per_value=32) == 3 * 32 * 32 * 32


def test_input_dim_values_matches_known_datasets():
    assert input_dim_values("mnist") == 1 * 28 * 28
    assert input_dim_values("cifar10") == 3 * 32 * 32


def test_raw_input_bits_per_sample_uses_32_bits_by_default():
    assert raw_input_bits_per_sample("cifar10") == 3 * 32 * 32 * 32


def test_latent_bits_per_sample_scales_linearly_with_dim():
    assert latent_bits_per_sample(16) == 16 * 32
    assert latent_bits_per_sample(64) == 4 * latent_bits_per_sample(16)


def test_total_raw_and_latent_bits_scale_with_num_samples():
    assert total_raw_bits("cifar10", num_samples=10) == raw_input_bits_per_sample("cifar10") * 10
    assert total_latent_bits(latent_dim=64, num_samples=10) == 64 * 32 * 10


def test_compression_ratio_matches_paper_example():
    # L=64 em CIFAR-10: 3*32*32 = 3072 valores crus vs 64 valores latentes -> CR = 48x
    raw_bits = raw_input_bits_per_sample("cifar10")
    compressed_bits = latent_bits_per_sample(64)
    assert compression_ratio(raw_bits, compressed_bits) == 48.0


def test_compression_ratio_handles_zero_compressed_bits():
    assert compression_ratio(raw_bits=100, compressed_bits=0) == 0.0


def test_model_update_bits_counts_all_parameters():
    model = nn.Linear(10, 2, bias=False)  # 20 parâmetros
    assert model_update_bits(model, bits_per_value=32) == 20 * 32


def test_model_update_bits_per_round_is_uplink_plus_downlink():
    model = nn.Linear(4, 1, bias=False)
    assert model_update_bits_per_round(model) == 2 * model_update_bits(model)
