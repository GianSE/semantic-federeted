"""Contabilidade de custo de comunicacao.

Duas contas distintas convivem neste sistema, e mistura-las inflaciona o ganho
reportado:

1. **Inferencia** -- o payload semantico transmitido por amostra. E aqui que a
   arquitetura ganha: um vetor latente de L dimensoes contra uma imagem inteira.

2. **Treinamento federado** -- os pesos trocados a cada rodada do FedAvg. Aqui a
   arquitetura semantica *perde*, por ter mais parametros (autoencoder +
   classificador) que o classificador do baseline. Esse custo independe do
   tamanho do latente e nao e reduzido pela compressao semantica.

Nota sobre a imagem bruta: MNIST e CIFAR-10 sao armazenados em uint8, logo o
baseline correto usa 8 bits por pixel. Contar 32 bits/pixel (float apos a
normalizacao) inflaria a razao de compressao em 4x.
"""

from typing import Optional, Tuple

import torch


MNIST_SHAPE = (1, 28, 28)
CIFAR10_SHAPE = (3, 32, 32)

RAW_BITS_PER_PIXEL = 8  # uint8, formato nativo de armazenamento
MODEL_BITS_PER_PARAM = 32  # float32, formato de troca de pesos no FedAvg


def input_shape(dataset_name: str) -> Tuple[int, ...]:
    if dataset_name.lower() == "mnist":
        return MNIST_SHAPE
    if dataset_name.lower() == "cifar10":
        return CIFAR10_SHAPE
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def input_dim_values(dataset_name: str) -> int:
    shape = input_shape(dataset_name)
    return shape[0] * shape[1] * shape[2]


# --- Fase de inferencia: payload semantico ---------------------------------


def raw_bits_per_sample(dataset_name: str, bits_per_pixel: int = RAW_BITS_PER_PIXEL) -> int:
    return input_dim_values(dataset_name) * bits_per_pixel


def latent_bits_per_sample(latent_dim: int, latent_bits: int = 32) -> int:
    return latent_dim * latent_bits


def total_raw_bits(
    dataset_name: str,
    num_samples: int,
    bits_per_pixel: int = RAW_BITS_PER_PIXEL,
) -> int:
    return raw_bits_per_sample(dataset_name, bits_per_pixel) * num_samples


def total_latent_bits(latent_dim: int, num_samples: int, latent_bits: int = 32) -> int:
    return latent_bits_per_sample(latent_dim, latent_bits) * num_samples


def compression_ratio(raw_bits: int, compressed_bits: int) -> float:
    if compressed_bits <= 0:
        return 0.0
    return raw_bits / compressed_bits


def bandwidth_savings(raw_bits: int, compressed_bits: int) -> float:
    """Fracao de trafego economizada, em [0, 1]."""
    if raw_bits <= 0:
        return 0.0
    return 1.0 - compressed_bits / raw_bits


# --- Fase de treinamento: troca de pesos no FedAvg -------------------------


def model_update_bits(model: torch.nn.Module, bits_per_param: int = MODEL_BITS_PER_PARAM) -> int:
    return sum(p.numel() for p in model.parameters()) * bits_per_param


def total_training_bits(
    model: torch.nn.Module,
    num_clients: int,
    rounds: int,
    bits_per_param: int = MODEL_BITS_PER_PARAM,
) -> int:
    """Uplink (pesos locais) + downlink (modelo global), por cliente e rodada."""
    return 2 * model_update_bits(model, bits_per_param) * num_clients * rounds


def comm_summary(
    dataset_name: str,
    num_samples: int,
    model: torch.nn.Module,
    num_clients: int,
    rounds: int,
    latent_dim: Optional[int] = None,
    latent_bits: int = 32,
) -> dict:
    """Contabilidade completa, com as duas fases explicitamente separadas."""
    raw_per_sample = raw_bits_per_sample(dataset_name)
    if latent_dim is None:  # baseline: transmite a imagem bruta
        inference_per_sample = raw_per_sample
    else:
        inference_per_sample = latent_bits_per_sample(latent_dim, latent_bits)

    return {
        "inference_bits_per_sample": inference_per_sample,
        "inference_bits_total": inference_per_sample * num_samples,
        "raw_bits_per_sample": raw_per_sample,
        "compression_ratio": compression_ratio(raw_per_sample, inference_per_sample),
        "bandwidth_savings": bandwidth_savings(raw_per_sample, inference_per_sample),
        "model_params": sum(p.numel() for p in model.parameters()),
        "training_bits_total": total_training_bits(model, num_clients, rounds),
    }
