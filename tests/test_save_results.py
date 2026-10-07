import json

import pandas as pd

from semantic_federated.reporting.save_results import save_results


def test_save_results_creates_csv_and_json(tmp_path):
    out_dir = str(tmp_path)
    save_results([{"dataset": "mnist", "accuracy": 0.9}], out_dir, "experiment_results")

    df = pd.read_csv(tmp_path / "experiment_results.csv")
    assert df.to_dict("records") == [{"dataset": "mnist", "accuracy": 0.9}]

    with open(tmp_path / "experiment_results.json", encoding="utf-8") as f:
        records = json.load(f)
    assert records == [{"dataset": "mnist", "accuracy": 0.9}]


def test_save_results_accumulates_across_calls(tmp_path):
    out_dir = str(tmp_path)
    save_results([{"dataset": "mnist", "accuracy": 0.9}], out_dir, "experiment_results")
    save_results([{"dataset": "cifar10", "accuracy": 0.6}], out_dir, "experiment_results")

    df = pd.read_csv(tmp_path / "experiment_results.csv")
    assert df["dataset"].tolist() == ["mnist", "cifar10"]

    with open(tmp_path / "experiment_results.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 2
