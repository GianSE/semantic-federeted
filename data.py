import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _get_transforms(dataset_name: str):
    if dataset_name.lower() == "mnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    if dataset_name.lower() == "cifar10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_dataset(dataset_name: str, train: bool):
    transform = _get_transforms(dataset_name)
    if dataset_name.lower() == "mnist":
        return datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    if dataset_name.lower() == "cifar10":
        return datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_clients(
    dataset,
    num_clients: int,
    seed: int,
    train_fraction: float = 1.0,
) -> List[Subset]:
    """Particiona o dataset entre clientes de forma IID (embaralhamento aleatório).

    `train_fraction` < 1.0 subamostra o conjunto de treino antes da partição,
    permitindo rodadas rápidas em CPU sem alterar a estrutura do experimento.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if train_fraction < 1.0:
        keep = max(num_clients, int(round(num_samples * train_fraction)))
        indices = indices[:keep]

    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def get_federated_dataloaders(
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    test_batch_size: int,
    seed: int,
    train_fraction: float = 1.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: Optional[bool] = None,
) -> Tuple[List[DataLoader], DataLoader]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = _load_dataset(dataset_name, train=True)
    test_dataset = _load_dataset(dataset_name, train=False)

    client_subsets = split_clients(
        train_dataset,
        num_clients=num_clients,
        seed=seed,
        train_fraction=train_fraction,
    )

    loader_opts = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_opts["persistent_workers"] = (
            True if persistent_workers is None else persistent_workers
        )
        loader_opts["worker_init_fn"] = _seed_worker

    client_loaders = []
    for client_id, subset in enumerate(client_subsets):
        # Um gerador por cliente mantém o embaralhamento reprodutível e
        # independente do número de workers.
        generator = torch.Generator()
        generator.manual_seed(seed + client_id)
        client_loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                **loader_opts,
            )
        )

    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, **loader_opts
    )
    return client_loaders, test_loader
