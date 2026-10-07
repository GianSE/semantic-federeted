import pandas as pd

from semantic_federated.reporting.plot_results import (
    plot_accuracy_vs_compression,
    plot_accuracy_vs_latent_dim,
    plot_accuracy_vs_noise,
    plot_channel_comparison,
    plot_comm_cost_vs_latent_dim,
    plot_partition_comparison,
)


def _base_rows():
    rows = []
    for latent_dim, cr, bits, acc in [(16, 192.0, 25_600_000, 0.60), (64, 48.0, 102_400_000, 0.65)]:
        rows.append({
            "dataset": "cifar10",
            "latent_dim": latent_dim,
            "noise_level": 0.0,
            "channel_type": "awgn",
            "partition": "iid",
            "accuracy_baseline_mean": 0.68,
            "accuracy_compressed_mean": acc,
            "accuracy_compressed_std": 0.01,
            "compression_ratio": cr,
            "communication_cost_bits": bits,
        })
    return rows


def test_plot_accuracy_vs_compression_creates_file(tmp_path):
    summary = pd.DataFrame(_base_rows())
    plot_accuracy_vs_compression(summary, str(tmp_path))
    assert (tmp_path / "accuracy_vs_compression_ratio.png").exists()


def test_plot_accuracy_vs_latent_dim_creates_file_with_log_fit(tmp_path):
    summary = pd.DataFrame(_base_rows())
    plot_accuracy_vs_latent_dim(summary, str(tmp_path))
    assert (tmp_path / "accuracy_vs_latent_dim.png").exists()


def test_plot_comm_cost_vs_latent_dim_creates_file(tmp_path):
    summary = pd.DataFrame(_base_rows())
    plot_comm_cost_vs_latent_dim(summary, str(tmp_path))
    assert (tmp_path / "communication_cost_vs_latent_dim.png").exists()


def test_plot_accuracy_vs_noise_creates_file(tmp_path):
    rows = _base_rows()
    rows.append({**rows[1], "noise_level": 0.05, "accuracy_compressed_mean": 0.66})
    summary = pd.DataFrame(rows)
    plot_accuracy_vs_noise(summary, str(tmp_path))
    assert (tmp_path / "accuracy_vs_noise_level.png").exists()


def test_plot_partition_comparison_skips_when_only_iid_present(tmp_path):
    summary = pd.DataFrame(_base_rows())  # só partition="iid"
    plot_partition_comparison(summary, str(tmp_path))
    assert not (tmp_path / "accuracy_iid_vs_dirichlet.png").exists()


def test_plot_partition_comparison_creates_file_when_both_partitions_present(tmp_path):
    rows = _base_rows()
    dirichlet_row = {**rows[1], "partition": "dirichlet", "accuracy_compressed_mean": 0.60}
    rows.append(dirichlet_row)
    summary = pd.DataFrame(rows)
    plot_partition_comparison(summary, str(tmp_path))
    assert (tmp_path / "accuracy_iid_vs_dirichlet.png").exists()


def test_plot_channel_comparison_skips_when_only_awgn_present(tmp_path):
    summary = pd.DataFrame(_base_rows())  # só channel_type="awgn"
    plot_channel_comparison(summary, str(tmp_path))
    assert not (tmp_path / "accuracy_by_channel_type.png").exists()


def test_plot_channel_comparison_creates_file_when_multiple_channels_present(tmp_path):
    rows = _base_rows()
    rayleigh_row = {**rows[1], "channel_type": "rayleigh", "accuracy_compressed_mean": 0.58}
    rows.append(rayleigh_row)
    summary = pd.DataFrame(rows)
    plot_channel_comparison(summary, str(tmp_path))
    assert (tmp_path / "accuracy_by_channel_type.png").exists()
