from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV_FILE = DATA_DIR / "collected_results.csv"
JSON_FILE = DATA_DIR / "collected_results.json"
NORMALIZED_DIR = DATA_DIR / "normalized"
THRESHOLD_DIR = DATA_DIR / "thresholds"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
REPORTS_DIR = DATA_DIR / "reports"
PLOTS_DIR = DATA_DIR / "plots"

def get_latest_snapshot(m):
    snapshot_dir = Path(SNAPSHOTS_DIR)
    csv_files = list(snapshot_dir.glob("snapshot_v*.csv"))
    if not csv_files:
        return "1"
    def _version_from_path(p: Path) -> int:
        parts = p.stem.rsplit("_v", 1)
        if len(parts) != 2:
            return -1
        try:
            return int(parts[1])
        except ValueError:
            return -1

    latest_file = max(csv_files, key=_version_from_path)
    latest_version = _version_from_path(latest_file)
    if latest_version < 0:
        return "1"
    if m == 1:
        return f"{latest_version + 1}"
    if m == 2:
        return f"{latest_version}"
    return None