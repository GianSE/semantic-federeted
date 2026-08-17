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


class _RoundSTE(torch.autograd.Function):
    """Arredondamento com straight-through estimator.

    O arredondamento tem derivada nula em quase todo ponto, o que bloquearia o
    gradiente. O STE propaga o gradiente como se a operacao fosse a identidade.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_latent(
    z: torch.Tensor,
    latent_bits: Optional[int],
    clip: float = 3.0,
) -> torch.Tensor:
    """Quantizacao uniforme do latente em `latent_bits` bits por dimensao.

    Aplicada apos a normalizacao de potencia, de modo que a faixa util e
    conhecida: com E[|z_i|^2]=1, o recorte em +/-3 cobre ~99,8% da massa se o
    latente for aproximadamente gaussiano.

    Sobre `clip`: ha um compromisso entre erro de recorte (caudas) e erro de
    granularidade (passo). Medido em latente gaussiano normalizado, SQNR em dB:

        clip     8 bits   6 bits   4 bits
         3.0      35.5     30.0     18.7
         4.0      40.8     28.7     16.2

    Com poucos bits, `clip=3` e melhor (passo menor); com 8 bits ou mais, o
    recorte passa a dominar. Mantemos 3.0 como padrao porque, mesmo limitado a
    ~35 dB, o erro de quantizacao fica muito abaixo do ruido de canal na faixa
    de SNR avaliada (<= 20 dB), nao sendo o fator limitante.

    `None` ou >= 32 mantem precisao total (sem quantizacao).
    """
    if latent_bits is None or latent_bits >= 32:
        return z
    if latent_bits < 1:
        raise ValueError("latent_bits deve ser >= 1")

    levels = 2**latent_bits - 1
    step = (2.0 * clip) / levels
    z_clipped = z.clamp(-clip, clip)
    indices = _RoundSTE.apply((z_clipped + clip) / step)
    return indices * step - clip


def _to_complex(z: torch.Tensor) -> torch.Tensor:
    """Empareha dimensoes reais adjacentes em simbolos complexos."""
    if z.shape[1] % 2 != 0:
        raise ValueError(
            f"Canais com desvanecimento exigem dimensao latente par (recebido L={z.shape[1]}), "
            "pois o latente e mapeado em simbolos complexos."
        )
    return torch.complex(z[:, 0::2], z[:, 1::2])


def _to_real(z_complex: torch.Tensor) -> torch.Tensor:
    """Inversa de `_to_complex`, preservando a ordem [re0, im0, re1, im1, ...]."""
    return torch.stack([z_complex.real, z_complex.imag], dim=-1).reshape(z_complex.shape[0], -1)


def _sample_fading(
    batch_size: int,
    k_factor: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Coeficiente de canal h com E[|h|^2] = 1, um por amostra (block fading).

    `k_factor` e o fator K de Rice em escala linear:
      - None  -> Rayleigh puro (sem componente de linha de visada)
      - K > 0 -> Rician com K = P_LOS / P_dispersa
      - K -> infinito recupera o canal AWGN (h = 1)

    O desvanecimento e por amostra, e nao por simbolo: um vetor latente e um
    pacote curto, transmitido dentro de um mesmo tempo de coerencia.
    """
    scatter_std = (0.5) ** 0.5  # CN(0,1): variancia 1/2 em cada componente
    shape = (batch_size, 1)
    scatter = torch.complex(
        torch.randn(shape, device=device, dtype=dtype) * scatter_std,
        torch.randn(shape, device=device, dtype=dtype) * scatter_std,
    )
    if k_factor is None:
        return scatter
    los_gain = (k_factor / (k_factor + 1.0)) ** 0.5
    scatter_gain = (1.0 / (k_factor + 1.0)) ** 0.5
    return los_gain + scatter_gain * scatter


def fading(
    z: torch.Tensor,
    snr_db: Optional[float],
    k_factor: Optional[float] = None,
) -> torch.Tensor:
    """Canal com desvanecimento plano e equalizacao com CSI perfeita.

        y = h * s + n        ->        s_hat = y / h = s + n / h

    Com CSI perfeita no receptor o sinal e recuperado, mas o ruido e amplificado
    por 1/|h|: em desvanecimentos profundos (|h| pequeno) a SNR instantanea
    despenca. E dai que vem a diferenca de comportamento em relacao ao AWGN.

    A convencao de SNR e a mesma do canal AWGN: com E[|z_i|^2] = 1 por dimensao
    real, cada componente do ruido tem desvio sigma = 10^(-SNR_dB/20).
    """
    sigma = snr_to_sigma(snr_db)
    symbols = _to_complex(z)
    h = _sample_fading(symbols.shape[0], k_factor, z.device, z.dtype)

    if sigma > 0:
        noise = torch.complex(
            torch.randn_like(symbols.real) * sigma,
            torch.randn_like(symbols.imag) * sigma,
        )
    else:
        noise = torch.zeros_like(symbols)

    received = h * symbols + noise
    equalized = received / h  # CSI perfeita
    return _to_real(equalized)


def apply_channel(
    z: torch.Tensor,
    snr_db: Optional[float],
    channel: str = "awgn",
    rician_k_db: Optional[float] = None,
) -> torch.Tensor:
    """Aplica o canal ao latente ja normalizado em potencia."""
    if channel == "awgn":
        return awgn(z, snr_db)
    if channel == "rayleigh":
        return fading(z, snr_db, k_factor=None)
    if channel == "rician":
        if rician_k_db is None:
            raise ValueError("Canal 'rician' exige --rician-k-db")
        return fading(z, snr_db, k_factor=10.0 ** (rician_k_db / 10.0))
    raise ValueError(f"Canal nao suportado: {channel}")


def apply_dropout_noise(latent: torch.Tensor, dropout_p: float, training: bool = True) -> torch.Tensor:
    if dropout_p <= 0:
        return latent
    return torch.nn.functional.dropout(latent, p=dropout_p, training=training)
