import pandas as pd

from semantic_federated.reporting.aggregate import aggregate_over_seeds


def test_aggregate_over_seeds_computes_mean_and_std_per_config(tmp_path):
    records = [
        {
            "dataset": "cifar10",
            "latent_dim": 64,
            "noise_level": 0.05,
            "channel_type": "awgn",
            "partition": "iid",
            "baseline_comm_mode": None,
            "accuracy_baseline": None,
            "accuracy_compressed": 0.60,
            "classification_loss": 1.0,
            "reconstruction_loss": 0.1,
            "seed": 1,
        },
        {
            "dataset": "cifar10",
            "latent_dim": 64,
            "noise_level": 0.05,
            "channel_type": "awgn",
            "partition": "iid",
            "baseline_comm_mode": None,
            "accuracy_baseline": None,
            "accuracy_compressed": 0.62,
            "classification_loss": 0.9,
            "reconstruction_loss": 0.1,
            "seed": 2,
        },
    ]
    csv_path = tmp_path / "experiment_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    summary = aggregate_over_seeds(str(csv_path), str(tmp_path / "tables"))
    row = summary.iloc[0]

    assert row["accuracy_compressed_mean"] == 0.61
    assert row["accuracy_compressed_count"] == 2
    assert (tmp_path / "tables" / "results_summary.csv").exists()


def test_aggregate_over_seeds_keeps_baseline_rows_separate_from_compressed(tmp_path):
    records = [
        {
            "dataset": "cifar10",
            "latent_dim": None,
            "noise_level": 0.0,
            "channel_type": None,
            "partition": "iid",
            "baseline_comm_mode": "raw",
            "accuracy_baseline": 0.68,
            "accuracy_compressed": None,
            "classification_loss": 1.2,
            "reconstruction_loss": None,
            "seed": 1,
        },
    ]
    csv_path = tmp_path / "experiment_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    summary = aggregate_over_seeds(str(csv_path), str(tmp_path / "tables"))
    assert len(summary) == 1
    assert summary.iloc[0]["accuracy_baseline_mean"] == 0.68
