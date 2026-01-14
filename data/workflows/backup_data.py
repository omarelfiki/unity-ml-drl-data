import os
import pandas as pd
import json

from paths import CSV_FILE, JSON_FILE

df = pd.read_csv(CSV_FILE, dtype=str)
df = df.fillna("")
df = df.astype(str)
data = df.to_dict(orient="records")

with open(JSON_FILE, "w") as jf:
    json.dump(data, jf, indent=4, ensure_ascii=False)

print(f"Converted '{CSV_FILE}' to '{JSON_FILE}' successfully.")
print(f"Total records: {len(data)}")