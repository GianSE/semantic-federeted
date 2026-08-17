"""Armazenamento de resultados por execução (run) e exportação agregada.

Cada configuração experimental gera um arquivo JSON próprio em `results/runs/`,
nomeado por um hash determinístico da configuração. Isso torna a grade de
experimentos *retomável*: rodar novamente pula o que já existe, e os CSVs finais
são sempre reconstruídos a partir dos runs em disco (nunca por append), de modo
que executar o pipeline duas vezes produz exatamente o mesmo resultado.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd


# Chaves que descrevem *como* a execução foi feita, e não *o que* foi executado.
# Ficam de fora do hash para que o mesmo experimento rodado na CPU local e numa
# GPU remota seja reconhecido como o mesmo run (evita refazer a grade inteira ao
# trocar de máquina). O dispositivo usado é registrado na saída para auditoria.
EXECUTION_KEYS = {
    "device",
    "num_workers",
    "show_progress",
    "force",
    "runs_dir",
    "out_dir",
}


def _canonical_config(config: Dict) -> Dict:
    return {k: v for k, v in sorted(config.items()) if k not in EXECUTION_KEYS}


def run_id(config: Dict) -> str:
    payload = json.dumps(_canonical_config(config), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def run_path(runs_dir: str, config: Dict) -> str:
    return os.path.join(runs_dir, f"{run_id(config)}.json")


def run_exists(runs_dir: str, config: Dict) -> bool:
    return os.path.isfile(run_path(runs_dir, config))


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def save_run(
    runs_dir: str,
    config: Dict,
    metrics: Dict,
    history: List[Dict],
    device: str = "unknown",
) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    record = {
        "run_id": run_id(config),
        "config": _canonical_config(config),
        "metrics": metrics,
        "history": history,
        "device": device,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = run_path(runs_dir, config)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return path


def load_runs(runs_dir: str) -> List[Dict]:
    if not os.path.isdir(runs_dir):
        return []
    records = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(runs_dir, name), "r", encoding="utf-8") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                print(f"[aviso] run corrompido, ignorado: {name}")
    return records


def _meta_columns(record: Dict) -> Dict:
    return {
        "run_id": record.get("run_id"),
        "device": record.get("device"),
        "git_commit": record.get("git_commit"),
        "timestamp": record.get("timestamp"),
    }


def results_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append({**record.get("config", {}), **record.get("metrics", {}), **_meta_columns(record)})
    return pd.DataFrame(rows)


def history_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        config = record.get("config", {})
        for entry in record.get("history", []):
            rows.append({**config, **entry, "run_id": record.get("run_id")})
    return pd.DataFrame(rows)


def export_results(
    runs_dir: str,
    out_dir: str,
    base_name: str = "experiment_results",
) -> Optional[pd.DataFrame]:
    """Reconstrói os CSVs agregados a partir dos runs em disco."""
    records = load_runs(runs_dir)
    if not records:
        print(f"[aviso] nenhum run encontrado em {runs_dir}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    results_df = results_dataframe(records)
    results_df.to_csv(os.path.join(out_dir, f"{base_name}.csv"), index=False)
    with open(os.path.join(out_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    history_df = history_dataframe(records)
    if not history_df.empty:
        history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    print(f"[ok] {len(records)} run(s) exportado(s) para {out_dir}")
    return results_df
