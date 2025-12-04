import os
import pandas as pd
import json

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/combined_results.csv"))
json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/combined_results.json"))

df = pd.read_csv(csv_path, dtype=str)
df = df.fillna("")
df = df.astype(str)
data = df.to_dict(orient="records")

with open(json_path, "w") as jf:
    json.dump(data, jf, indent=4, ensure_ascii=False)

print(f"Converted '{csv_path}' to '{json_path}' successfully.")
print(f"Total records: {len(data)}")