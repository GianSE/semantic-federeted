import random
from typing import List, Tuple

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


def split_clients(
    dataset, num_clients: int, seed: int
) -> List[Subset]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def _get_targets(dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset has no 'targets' attribute; required for Dirichlet partitioning")
    return np.asarray(targets)


def split_clients_dirichlet(
    dataset, num_clients: int, alpha: float, seed: int
) -> List[Subset]:
    """Particiona por rótulo via distribuição de Dirichlet (não-IID).

    Para cada classe, a fração de amostras destinada a cada cliente é amostrada
    de Dirichlet(alpha, ..., alpha). alpha baixo (ex.: 0.1) concentra cada classe
    em poucos clientes; alpha alto (ex.: 100) se aproxima do particionamento IID.
    Ver Hsu et al. (2019) para a referência padrão desse esquema.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    targets = _get_targets(dataset)
    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(class_indices, split_points)):
            client_indices[client_id].extend(split.tolist())

    for indices in client_indices:
        rng.shuffle(indices)

    return [Subset(dataset, indices) for indices in client_indices]


def get_federated_dataloaders(
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    test_batch_size: int,
    seed: int,
    num_workers: int = 0,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
) -> Tuple[List[DataLoader], DataLoader]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = _load_dataset(dataset_name, train=True)
    test_dataset = _load_dataset(dataset_name, train=False)

    if partition == "iid":
        client_subsets = split_clients(train_dataset, num_clients=num_clients, seed=seed)
    elif partition == "dirichlet":
        client_subsets = split_clients_dirichlet(
            train_dataset, num_clients=num_clients, alpha=dirichlet_alpha, seed=seed
        )
    else:
        raise ValueError(f"Unsupported partition strategy: {partition}")

    client_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for subset in client_subsets
    ]
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    return client_loaders, test_loader
