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
