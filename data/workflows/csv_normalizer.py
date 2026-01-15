import pandas as pd
from paths import get_latest_snapshot, NORMALIZED_DIR
from datetime import datetime

INPUT_CSV = get_latest_snapshot()
timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
OUTPUT_CSV = NORMALIZED_DIR / f"normalized_results_{timestamp}.csv"

# Metrics to z-score
METRIC_COLS = [
    "reward_mean",
    "early_reward_mean",
    "final_reward_mean",
    "best_reward",
    "p_loss_mean",
    "v_loss_mean",
    "entropy_mean",
]

def main():
    print(f"[INFO] Loading data from {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        raise ValueError(f"[ERROR]: Could not load CSV: {e}")

    if "environment" not in df.columns:
        raise ValueError("[ERROR]: Column 'environment' not found in CSV.")

    # Enforce numeric types for chosen metrics
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    norm_df = df.copy()

    # Per-env z-score
    for env, env_df in norm_df.groupby("environment"):
        print(f"[INFO] Standardizing environment: {env}")
        idx = env_df.index

        for col in METRIC_COLS:
            if col not in norm_df.columns:
                continue

            col_vals = norm_df.loc[idx, col]
            mean = col_vals.mean(skipna=True)
            std = col_vals.std(ddof=0, skipna=True)

            if pd.isna(std) or std == 0:
                print(f"[WARNING] Skipping {col} for {env} (std is 0 or NaN).")
                continue

            z = (col_vals - mean) / std
            norm_df.loc[idx, col] = z

    norm_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Per-environment z-scored results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()