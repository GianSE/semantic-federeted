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


def get_federated_dataloaders(
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    test_batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> Tuple[List[DataLoader], DataLoader]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = _load_dataset(dataset_name, train=True)
    test_dataset = _load_dataset(dataset_name, train=False)

    client_subsets = split_clients(train_dataset, num_clients=num_clients, seed=seed)
    client_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for subset in client_subsets
    ]
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    return client_loaders, test_loader
