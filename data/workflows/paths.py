from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV_FILE = DATA_DIR / "collected_results.csv"
JSON_FILE = DATA_DIR / "collected_results.json"
NORMALIZED_CSV = DATA_DIR / "normalized_results.csv"
THRESHOLD_DIR = DATA_DIR / "thresholds"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
REPORTS_DIR = DATA_DIR / "reports"
PLOTS_DIR = DATA_DIR / "plots"