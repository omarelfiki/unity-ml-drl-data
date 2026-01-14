"""
Computing empirical reward thresholds per environment by parsing the CSV dataset.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime

from paths import CSV_FILE, THRESHOLD_DIR

# Default Parameters
TAIL_STEPS_DEFAULT = 50_000
ALPHA_DEFAULT = 0.8

def compute_threshold_from_rewards(values, tail_steps=TAIL_STEPS_DEFAULT, alpha=ALPHA_DEFAULT):
    """Compute R* (90th percentile of tail) and threshold T = α × R*."""
    if not values:
        return None, None
    tail = values[-tail_steps:] if len(values) > tail_steps else values
    r_star = float(np.percentile(tail, 90))
    t_run = float(alpha * r_star)
    return r_star, t_run

def find_first_step_reaching_threshold(steps, rewards, threshold):
    """Return the first step where reward ≥ threshold, else None."""
    if not steps or threshold is None:
        return None
    for s, r in zip(steps, rewards):
        if r >= threshold:
            return s
    return None

def main():
    if not CSV_FILE.exists():
        print(f"Error: Data file not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    if not {'environment', 'steps', 'reward_mean'}.issubset(df.columns):
        print("Error: CSV missing required columns 'environment', 'steps', or 'reward_mean'")
        return

    thresholds = {}

    for env_name, group in df.groupby("environment"):
        # Sort by step ascending and align steps with non-null rewards
        group_sorted = group.sort_values("steps")
        rewards_series = pd.to_numeric(group_sorted["reward_mean"], errors="coerce").dropna()
        steps_series = group_sorted.loc[rewards_series.index, "steps"]

        rewards = rewards_series.tolist()
        steps = steps_series.tolist()

        r_star, t_run = compute_threshold_from_rewards(rewards)
        if r_star is None:
            continue

        first_reach = find_first_step_reaching_threshold(steps, rewards, t_run)
        best_reward = max(rewards) if rewards else None
        best_step = steps[int(np.argmax(rewards))] if rewards else None

        thresholds[env_name] = {
            "environment": env_name,
            "R_star": r_star,
            "T_run": t_run,
            "alpha": ALPHA_DEFAULT,
            "method": "empirical_reference",
            "window_last_steps": TAIL_STEPS_DEFAULT,
            "reward_points": len(rewards),
            "best_reward": best_reward,
            "best_reward_step": best_step,
            "first_step_reaching_threshold": first_reach,
        }

        print(f"{env_name}: R*={r_star:.3f}, T={t_run:.3f}, best={best_reward:.3f}, "
              f"points={len(rewards)}, first reach step={first_reach}")

    version_ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    payload = {
        "version": version_ts,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(CSV_FILE.resolve()),
        "method": "empirical_reference",
        "alpha": ALPHA_DEFAULT,
        "window_last_steps": TAIL_STEPS_DEFAULT,
        "thresholds": thresholds,
    }

    # Ensure output directory exists
    THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Write timestamped and "latest" files
    timestamped_path = THRESHOLD_DIR / f"thresholds_{version_ts}.json"
    latest_path = THRESHOLD_DIR / "latest_thresholds.json"

    with open(timestamped_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved thresholds version to: {timestamped_path}")
    print(f"Updated latest thresholds at: {latest_path}")

if __name__ == "__main__":
    main()