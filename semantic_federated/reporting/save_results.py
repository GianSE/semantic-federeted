import json
import os
from typing import List, Dict

import pandas as pd


def save_results(records: List[Dict], out_dir: str, base_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    json_path = os.path.join(out_dir, f"{base_name}.json")
    
    df = pd.DataFrame(records)

    csv_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    if not csv_exists:
        df.to_csv(csv_path, index=False)
    else:
        # Caminho rapido: mesmo esquema de colunas (ex.: varias configs comprimidas em
        # sequencia) -> so acrescenta linhas, sem reler o arquivo inteiro.
        existing_header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if list(df.columns) == existing_header:
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            # Esquema mudou (ex.: baseline sem 'checkpoint_path' vs. comprimido com) --
            # reconcilia as colunas relendo o CSV, em vez de gerar linhas com numero
            # de campos diferente (CSV invalido).
            old_df = pd.read_csv(csv_path)
            pd.concat([old_df, df], ignore_index=True).to_csv(csv_path, index=False)
    
    # Accumulate in JSON
    all_records = []
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_records = json.load(f)
        except json.JSONDecodeError:
            pass
    all_records.extend(records)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
