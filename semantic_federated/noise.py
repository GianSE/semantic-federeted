import torch


def add_gaussian_noise(latent: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return latent
    noise = torch.randn_like(latent) * sigma
    return latent + noise


def apply_dropout_noise(latent: torch.Tensor, dropout_p: float, training: bool = True) -> torch.Tensor:
    if dropout_p <= 0:
        return latent
    return torch.nn.functional.dropout(latent, p=dropout_p, training=training)


def _rayleigh_magnitude(batch_shape, device) -> torch.Tensor:
    # h = (a + jb)/sqrt(2), com a, b ~ N(0, 1) i.i.d. -> |h| segue Rayleigh normalizada
    # para ganho medio de potencia unitario: E[|h|^2] = 1 (convencao usual em canais sem fio).
    a = torch.randn(batch_shape, device=device)
    b = torch.randn(batch_shape, device=device)
    return torch.sqrt(a**2 + b**2) / torch.sqrt(torch.tensor(2.0, device=device))


def apply_rayleigh_fading(latent: torch.Tensor, scale: float, noise_sigma: float = 0.0) -> torch.Tensor:
    """Desvanecimento Rayleigh (sem linha de visada): z~ = h*z + n, h ~ Rayleigh(scale) por amostra."""
    if scale <= 0:
        return add_gaussian_noise(latent, noise_sigma)
    batch_shape = (latent.shape[0],) + (1,) * (latent.dim() - 1)
    h = _rayleigh_magnitude(batch_shape, latent.device) * scale
    return add_gaussian_noise(latent * h, noise_sigma)


def apply_rician_fading(
    latent: torch.Tensor, k_factor: float, scale: float, noise_sigma: float = 0.0
) -> torch.Tensor:
    """Desvanecimento Rician (com linha de visada): h = sqrt(K/(K+1)) + sqrt(1/(K+1))*Rayleigh.

    k_factor alto -> canal dominado pelo componente de linha de visada (pouco desvanecimento).
    k_factor -> 0 reduz ao caso Rayleigh puro.
    """
    if scale <= 0:
        return add_gaussian_noise(latent, noise_sigma)
    if k_factor < 0:
        raise ValueError("k_factor must be non-negative")
    batch_shape = (latent.shape[0],) + (1,) * (latent.dim() - 1)
    los_term = (k_factor / (k_factor + 1)) ** 0.5
    nlos_scale = (1.0 / (k_factor + 1)) ** 0.5
    h = (los_term + nlos_scale * _rayleigh_magnitude(batch_shape, latent.device)) * scale
    return add_gaussian_noise(latent * h, noise_sigma)


def apply_channel(
    latent: torch.Tensor,
    channel_type: str,
    noise_sigma: float = 0.0,
    fading_scale: float = 1.0,
    rician_k: float = 1.0,
) -> torch.Tensor:
    """Despacha para o modelo de canal escolhido. 'awgn' preserva o comportamento original."""
    if channel_type == "awgn":
        return add_gaussian_noise(latent, noise_sigma)
    if channel_type == "rayleigh":
        return apply_rayleigh_fading(latent, scale=fading_scale, noise_sigma=noise_sigma)
    if channel_type == "rician":
        return apply_rician_fading(latent, k_factor=rician_k, scale=fading_scale, noise_sigma=noise_sigma)
    raise ValueError(f"Unsupported channel_type: {channel_type}")
