import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from metrics import average_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def perturb_state_dict(
    state_dict: Dict[str, torch.Tensor],
    snr_db: Optional[float],
) -> Dict[str, torch.Tensor]:
    """Aplica ruido de canal a uma atualizacao de modelo antes da agregacao.

    O FedAvg pressupoe agregacao aritmetica exata, isto e, um enlace ideal para
    os pesos. Numa rede real as atualizacoes tambem atravessam um canal ruidoso,
    e o erro se propaga direto para o modelo global.

    Como cada tensor de parametros tem sua propria escala, a SNR e definida por
    tensor: o desvio do ruido e proporcional ao RMS do proprio tensor, de modo
    que `snr_db` signifique a mesma condicao de canal em toda a rede. Buffers
    inteiros (contadores, indices) sao preservados: nao sao sinal transmitido.

    `None` representa enlace ideal e nao consome numeros aleatorios, de modo que
    o comportamento seja identico ao de antes desta funcionalidade existir.
    """
    if snr_db is None:
        return state_dict

    sigma_relative = 10.0 ** (-snr_db / 20.0)
    perturbed = {}
    for key, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            perturbed[key] = tensor
            continue
        rms = tensor.pow(2).mean().sqrt()
        perturbed[key] = tensor + torch.randn_like(tensor) * (sigma_relative * rms)
    return perturbed


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]):
    if not state_dicts:
        return {}
    avg_state = {}
    total_weight = float(sum(weights))
    for key in state_dicts[0].keys():
        avg_state[key] = sum(
            state_dict[key] * (weight / total_weight)
            for state_dict, weight in zip(state_dicts, weights)
        )
    return avg_state


def train_local(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    train_step_fn: Callable,
    device: torch.device,
    show_progress: bool,
) -> Dict[str, float]:
    model.train()
    batch_metrics = []
    batch_sizes = []
    batch_iter = loader
    if show_progress:
        batch_iter = tqdm(loader, desc="Batches", leave=False)
    for batch in batch_iter:
        optimizer.zero_grad()
        loss, metrics = train_step_fn(model, batch, device)
        loss.backward()
        optimizer.step()
        batch_metrics.append(metrics)
        batch_sizes.append(batch[1].size(0))
    return average_metrics(batch_metrics, batch_sizes)


def evaluate_model(
    model: nn.Module,
    loader,
    eval_step_fn: Callable,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    batch_metrics = []
    batch_sizes = []
    with torch.no_grad():
        for batch in loader:
            _, metrics = eval_step_fn(model, batch, device)
            batch_metrics.append(metrics)
            batch_sizes.append(batch[1].size(0))
    return average_metrics(batch_metrics, batch_sizes)


def federated_train(
    global_model: nn.Module,
    client_loaders: List,
    test_loader,
    rounds: int,
    local_epochs: int,
    optimizer_fn: Callable,
    train_step_fn: Callable,
    eval_step_fn: Callable,
    device: torch.device,
    weight_snr_db: Optional[float] = None,
    show_progress: bool = True,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    history = []

    round_iter = range(rounds)
    if show_progress:
        round_iter = tqdm(round_iter, desc="Federated Rounds")

    for round_idx in round_iter:
        client_states = []
        client_sizes = []
        round_metrics = []

        client_iter = enumerate(client_loaders)
        if show_progress:
            client_iter = tqdm(client_iter, total=len(client_loaders), desc="Clients", leave=False)

        for client_id, loader in client_iter:
            client_model = copy.deepcopy(global_model)
            optimizer = optimizer_fn(client_model.parameters())
            for _ in range(local_epochs):
                metrics = train_local(
                    client_model, loader, optimizer, train_step_fn, device, show_progress
                )
            round_metrics.append(metrics)
            # Uplink dos pesos: a atualizacao local atravessa o canal antes
            # de chegar ao servidor para agregacao.
            client_states.append(
                perturb_state_dict(client_model.state_dict(), weight_snr_db)
            )
            client_sizes.append(len(loader.dataset))

        avg_round_metrics = average_metrics(round_metrics)
        avg_state = average_state_dicts(client_states, client_sizes)
        global_model.load_state_dict(avg_state)
        eval_metrics = evaluate_model(global_model, test_loader, eval_step_fn, device)
        history.append(
            {
                "round": round_idx + 1,
                **avg_round_metrics,
                **{f"eval_{k}": v for k, v in eval_metrics.items()},
            }
        )

        if show_progress:
            loss_value = avg_round_metrics.get("loss")
            acc_value = avg_round_metrics.get("accuracy")
            if loss_value is not None and acc_value is not None:
                tqdm.write(f"Round {round_idx + 1}: loss={loss_value:.4f}, accuracy={acc_value:.4f}")
            elif loss_value is not None:
                tqdm.write(f"Round {round_idx + 1}: loss={loss_value:.4f}")

    return global_model, history
