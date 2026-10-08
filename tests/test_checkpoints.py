import torch
from torch import nn

from semantic_federated.reporting.checkpoints import (
    checkpoint_filename,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)


def _config(**overrides):
    base = {
        "dataset": "cifar10",
        "latent_dim": 64,
        "noise_level": 0.05,
        "channel_type": "awgn",
        "partition": "iid",
        "seed": 42,
    }
    base.update(overrides)
    return base


def test_checkpoint_filename_is_deterministic_and_readable():
    name = checkpoint_filename(_config())
    assert name == "cifar10_L64_noise0.05_awgn_iid_seed42.pt"


def test_checkpoint_filename_formats_zero_noise_without_trailing_zeros():
    name = checkpoint_filename(_config(noise_level=0.0))
    assert "noise0_" in name


def test_checkpoint_filename_differs_by_seed_and_latent_dim():
    name_a = checkpoint_filename(_config(seed=1))
    name_b = checkpoint_filename(_config(seed=2))
    name_c = checkpoint_filename(_config(latent_dim=16))
    assert name_a != name_b != name_c


def test_save_and_load_checkpoint_round_trip(tmp_path):
    model = nn.Linear(4, 2)
    config = _config()

    saved_path = save_checkpoint(model, config, out_dir=str(tmp_path))
    assert saved_path == checkpoint_path(config, str(tmp_path))

    record = load_checkpoint(saved_path)
    assert record["config"] == config

    reloaded = nn.Linear(4, 2)
    reloaded.load_state_dict(record["state_dict"])
    for p1, p2 in zip(model.parameters(), reloaded.parameters()):
        assert torch.equal(p1, p2)


def test_load_checkpoint_missing_file_raises_with_helpful_message(tmp_path):
    missing = str(tmp_path / "does_not_exist.pt")
    try:
        load_checkpoint(missing)
        assert False, "esperava FileNotFoundError"
    except FileNotFoundError as e:
        assert "nao encontrado" in str(e)
