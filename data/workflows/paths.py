import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV_FILE = DATA_DIR / "collected_results.csv"
JSON_FILE = DATA_DIR / "collected_results.json"
NORMALIZED_DIR = DATA_DIR / "normalized"
THRESHOLD_DIR = DATA_DIR / "thresholds"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
REPORTS_DIR = DATA_DIR / "reports"
PLOTS_DIR = DATA_DIR / "plots"

def get_latest_snapshot():
    snapshot_dir = Path(SNAPSHOTS_DIR)
    csv_files = list(snapshot_dir.glob("prediction_snapshot_*.csv"))
    if not csv_files:
        return None

    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file