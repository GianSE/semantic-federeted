"""Modelo de canal sobre o espaco latente.

Motivacao: sem restricao de potencia, a escala de `z` e livre e treinavel, de
modo que o codificador pode simplesmente aumentar ||z|| ate tornar qualquer
ruido de amplitude fixa irrelevante. Nesse regime um "sigma" absoluto nao define
uma condicao de canal -- so a razao entre potencias define.

Por isso o latente e normalizado para potencia unitaria por dimensao antes do
canal, e o nivel de ruido passa a ser parametrizado pela SNR em dB:

    E[|z_i|^2] = 1   =>   SNR = 1 / sigma^2   =>   sigma = 10^(-SNR_dB / 20)

Assim "SNR = 0 dB" significa potencia de sinal igual a de ruido, com o mesmo
sentido usado em comunicacoes.
"""

from typing import Optional

import torch


def normalize_power(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normaliza cada vetor latente para potencia unitaria por dimensao."""
    latent_dim = z.shape[1]
    norm = z.norm(dim=1, keepdim=True).clamp_min(eps)
    return z * (latent_dim**0.5) / norm


def snr_to_sigma(snr_db: Optional[float]) -> float:
    """Desvio-padrao do ruido para uma SNR em dB, assumindo potencia unitaria.

    `None` representa canal ideal (sem ruido).
    """
    if snr_db is None:
        return 0.0
    return float(10.0 ** (-snr_db / 20.0))


def sigma_to_snr_db(sigma: float) -> Optional[float]:
    """Inversa de `snr_to_sigma`. Util para reinterpretar experimentos antigos."""
    if sigma <= 0:
        return None
    return float(-20.0 * torch.log10(torch.tensor(sigma)).item())


MATCH_TRAIN = "match"


def parse_snr(value):
    """Converte um argumento de linha de comando em SNR.

    'none'/'ideal' -> None (canal ideal); 'match' -> sentinela que indica
    "mesma SNR do treino"; caso contrario, float em dB.
    """
    if value is None or isinstance(value, float):
        return value
    text = str(value).strip().lower()
    if text in ("none", "ideal", "inf"):
        return None
    if text == MATCH_TRAIN:
        return MATCH_TRAIN
    return float(text)


def resolve_test_snr(snr_test, snr_train):
    """Resolve a sentinela 'match' para a SNR de treino."""
    return snr_train if snr_test == MATCH_TRAIN else snr_test


def awgn(z: torch.Tensor, snr_db: Optional[float]) -> torch.Tensor:
    sigma = snr_to_sigma(snr_db)
    if sigma <= 0:
        return z
    return z + torch.randn_like(z) * sigma


def apply_channel(
    z: torch.Tensor,
    snr_db: Optional[float],
    channel: str = "awgn",
) -> torch.Tensor:
    """Aplica o canal ao latente ja normalizado em potencia.

    Fases seguintes do projeto acrescentam desvanecimento (Rayleigh/Rician)
    aqui, mantendo a mesma interface.
    """
    if channel == "awgn":
        return awgn(z, snr_db)
    raise ValueError(f"Canal nao suportado: {channel}")


def apply_dropout_noise(latent: torch.Tensor, dropout_p: float, training: bool = True) -> torch.Tensor:
    if dropout_p <= 0:
        return latent
    return torch.nn.functional.dropout(latent, p=dropout_p, training=training)
