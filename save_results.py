import json
import os
from typing import List, Dict

import pandas as pd


def save_results(records: List[Dict], out_dir: str, base_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    json_path = os.path.join(out_dir, f"{base_name}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
