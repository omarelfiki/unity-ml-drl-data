from pathlib import Path
import pandas as pd
from paths import CSV_FILE, SNAPSHOTS_DIR

OUTPUT_CSV = SNAPSHOTS_DIR / "prediction_snapshot.csv"

threshold_cols = [
    "threshold_value",
    "steps_to_threshold",
    "time_to_threshold",
    "threshold_version",
    "run_reached_threshold",
]

def filter_runs_with_threshold(input_path: Path, output_path: Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    missing = [c for c in threshold_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # keep rows with any non-null threshold info
    mask = df[threshold_cols].notna().any(axis=1)
    df_filtered = df[mask]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)

if __name__ == "__main__":
    filter_runs_with_threshold(CSV_FILE, OUTPUT_CSV)
