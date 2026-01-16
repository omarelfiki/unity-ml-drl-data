from pathlib import Path
import pandas as pd
from paths import CSV_FILE, SNAPSHOTS_DIR
from paths import get_latest_snapshot

# Timestamped filename
version = get_latest_snapshot(1)
OUTPUT_CSV = SNAPSHOTS_DIR / f"snapshot_v{version}.csv"

threshold_defaults = {
    "3DBall": 2.56576,
    "Basic": 0.6096,
    "BigWallJump": -0.864552,
    "Crawler": -0.79424,
    "Hallway": -0.556176,
    "PushBlock": 2.46952,
}

threshold_cols = [
    "threshold_value","steps_to_threshold","time_to_threshold",
    "threshold_version","run_reached_threshold",
]

def define_thresholds(row):
    if pd.isna(row.get("threshold_value")):
        env = row.get("environment")
        if env in threshold_defaults:
            return threshold_defaults[env]
        print(f"[WARNING] No threshold configured for environment '{env}'")
        return row.get("threshold_value")
    return row.get("threshold_value")

def identify_success(row):
    steps_val = row.get("steps_to_threshold")
    thr_val = row.get("threshold_value")
    final_mean = row.get("final_reward_mean")

    if pd.notna(steps_val):
        return 1
    if pd.notna(final_mean) and pd.notna(thr_val):
        return int(final_mean >= thr_val)
    return 0

def filter_runs_with_threshold(input_path: Path, output_path: Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    missing = [c for c in threshold_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR]: Missing expected columns in CSV: {missing}")

    # normalize threshold-related columns
    df["threshold_value"] = df.apply(define_thresholds, axis=1)

    if "steps_to_threshold" in df.columns:
        df["steps_to_threshold"] = pd.to_numeric(
            df["steps_to_threshold"], errors="coerce"
        )
    if "time_to_threshold" in df.columns:
        df["time_to_threshold"] = pd.to_numeric(
            df["time_to_threshold"], errors="coerce"
        )

    df["run_reached_threshold"] = df.apply(identify_success, axis=1)

    # keep rows with actual evaluated threshold outcome
    mask = df["steps_to_threshold"].notna() | df["time_to_threshold"].notna()
    df_filtered = df[mask]

    # columns to keep for prediction
    columns_to_keep = [
        "run_id","environment","seed","algorithm","learning_rate","steps",
        "batch_size","buffer_size","epochs","num_agents","total_time","average_cpu",
        "average_ram","early_reward_mean","early_reward_mean_step","final_reward_mean",
        "final_reward_mean_step","best_reward","best_reward_step","p_loss_mean",
        "p_loss_mean_step","v_loss_mean","v_loss_mean_step","entropy_mean",
        "entropy_mean_step","step_interval","threshold_value","steps_to_threshold",
        "time_to_threshold","threshold_version","run_reached_threshold",
    ]
    columns_to_keep = [c for c in columns_to_keep if c in df_filtered.columns]

    snapshot_df = df_filtered[columns_to_keep]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]: Writing filtered CSV to: {output_path}")
    if not output_path.exists():
        snapshot_df.to_csv(output_path, index=False)
    else:
        print("[WARNING]: Output file already exists, skipping write.")

if __name__ == "__main__":
    filter_runs_with_threshold(CSV_FILE, OUTPUT_CSV)