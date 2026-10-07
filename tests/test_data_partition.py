import numpy as np

from semantic_federated.data import split_clients_dirichlet


class _DummyDataset:
    """Dataset mínimo (sem torch) só para exercitar o particionamento por rótulo."""

    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return idx, self.targets[idx]


def _make_balanced_dataset(num_classes=4, samples_per_class=100):
    targets = []
    for class_id in range(num_classes):
        targets.extend([class_id] * samples_per_class)
    return _DummyDataset(targets)


def test_split_clients_dirichlet_covers_every_sample_exactly_once():
    dataset = _make_balanced_dataset()
    subsets = split_clients_dirichlet(dataset, num_clients=5, alpha=0.5, seed=0)

    all_indices = sorted(idx for subset in subsets for idx in subset.indices)
    assert all_indices == list(range(len(dataset)))


def test_split_clients_dirichlet_high_alpha_is_roughly_balanced():
    dataset = _make_balanced_dataset(num_classes=4, samples_per_class=200)
    subsets = split_clients_dirichlet(dataset, num_clients=4, alpha=100.0, seed=0)
    sizes = [len(subset) for subset in subsets]
    # alpha alto deve aproximar o IID: nenhum cliente deve ficar muito acima/abaixo da media.
    expected = len(dataset) / 4
    for size in sizes:
        assert abs(size - expected) < expected * 0.3


def test_split_clients_dirichlet_low_alpha_concentrates_classes():
    dataset = _make_balanced_dataset(num_classes=2, samples_per_class=200)
    subsets = split_clients_dirichlet(dataset, num_clients=4, alpha=0.01, seed=0)
    sizes = np.array([len(subset) for subset in subsets])
    # alpha baixo deve concentrar fortemente as classes em poucos clientes,
    # gerando tamanhos de cliente muito desiguais (alguns quase vazios).
    assert sizes.max() > sizes.mean() * 1.5


def test_split_clients_dirichlet_rejects_invalid_alpha():
    dataset = _make_balanced_dataset()
    try:
        split_clients_dirichlet(dataset, num_clients=3, alpha=0.0, seed=0)
        assert False, "esperava ValueError para alpha <= 0"
    except ValueError:
        pass
