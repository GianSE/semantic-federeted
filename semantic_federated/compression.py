from typing import Tuple

import torch


MNIST_SHAPE = (1, 28, 28)
CIFAR10_SHAPE = (3, 32, 32)


def bits_for_shape(shape: Tuple[int, ...], bits_per_value: int = 32) -> int:
    num_values = 1
    for dim in shape:
        num_values *= dim
    return num_values * bits_per_value


def input_dim_values(dataset_name: str) -> int:
    if dataset_name.lower() == "mnist":
        return MNIST_SHAPE[0] * MNIST_SHAPE[1] * MNIST_SHAPE[2]
    if dataset_name.lower() == "cifar10":
        return CIFAR10_SHAPE[0] * CIFAR10_SHAPE[1] * CIFAR10_SHAPE[2]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def raw_input_bits_per_sample(dataset_name: str, bits_per_value: int = 32) -> int:
    if dataset_name.lower() == "mnist":
        return bits_for_shape(MNIST_SHAPE, bits_per_value)
    if dataset_name.lower() == "cifar10":
        return bits_for_shape(CIFAR10_SHAPE, bits_per_value)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def latent_bits_per_sample(latent_dim: int, bits_per_value: int = 32) -> int:
    return latent_dim * bits_per_value


def model_update_bits(model: torch.nn.Module, bits_per_value: int = 32) -> int:
    total_params = sum(param.numel() for param in model.parameters())
    return total_params * bits_per_value


def raw_input_bits_total(
    dataset_name: str,
    num_samples: int,
    bits_per_value: int = 32,
) -> int:
    return raw_input_bits_per_sample(dataset_name, bits_per_value) * num_samples


def latent_bits_total(
    latent_dim: int,
    num_samples: int,
    bits_per_value: int = 32,
) -> int:
    return latent_bits_per_sample(latent_dim, bits_per_value) * num_samples


def total_raw_bits(dataset_name: str, num_samples: int, bits_per_value: int = 32) -> int:
    return input_dim_values(dataset_name) * num_samples * bits_per_value


def total_latent_bits(latent_dim: int, num_samples: int, bits_per_value: int = 32) -> int:
    return latent_dim * num_samples * bits_per_value


def model_update_bits_per_round(
    model: torch.nn.Module,
    bits_per_value: int = 32,
) -> int:
    # Uplink + downlink payload for full model update.
    return model_update_bits(model, bits_per_value) * 2


def compression_ratio(raw_bits: int, compressed_bits: int) -> float:
    if compressed_bits <= 0:
        return 0.0
    return raw_bits / compressed_bits
