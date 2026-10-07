import json
import os
from typing import List, Dict

import pandas as pd


def save_results(records: List[Dict], out_dir: str, base_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    json_path = os.path.join(out_dir, f"{base_name}.json")
    
    df = pd.DataFrame(records)

    # Append directly instead of reading the whole CSV back on every call.
    csv_has_header = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    df.to_csv(csv_path, mode="a", header=not csv_has_header, index=False)
    
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
