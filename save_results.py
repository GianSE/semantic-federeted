import json
import os
from typing import List, Dict

import pandas as pd


def save_results(records: List[Dict], out_dir: str, base_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    json_path = os.path.join(out_dir, f"{base_name}.json")
    
    df = pd.DataFrame(records)
    
    # Accumulate in CSV handling new headers
    if os.path.isfile(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            df = pd.concat([old_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
            
    df.to_csv(csv_path, index=False)
    
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
