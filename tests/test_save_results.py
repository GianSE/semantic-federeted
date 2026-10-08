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


def test_save_results_reconciles_when_columns_differ_across_calls(tmp_path):
    """Regressao: baseline e comprimido tem esquemas diferentes (ex.: checkpoint_path
    so existe no comprimido) -- apendar direto geraria um CSV com numero de campos
    inconsistente entre linhas."""
    out_dir = str(tmp_path)
    save_results([{"dataset": "mnist", "accuracy_baseline": 0.9}], out_dir, "experiment_results")
    save_results(
        [{"dataset": "mnist", "accuracy_compressed": 0.8, "checkpoint_path": "foo.pt"}],
        out_dir,
        "experiment_results",
    )

    # Nao pode levantar ParserError por numero de campos inconsistente.
    df = pd.read_csv(tmp_path / "experiment_results.csv")
    assert len(df) == 2
    assert set(df.columns) == {"dataset", "accuracy_baseline", "accuracy_compressed", "checkpoint_path"}
    assert df.loc[0, "checkpoint_path"] != df.loc[0, "checkpoint_path"]  # NaN para a linha do baseline
    assert df.loc[1, "checkpoint_path"] == "foo.pt"
