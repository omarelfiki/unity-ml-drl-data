"""
Compute empirical reward thresholds per environment by parsing a shared CSV dataset.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "combined_results.csv"
OUTPUT_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "reference_thresholds.json"

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
    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    if not {'environment', 'steps', 'reward_mean'}.issubset(df.columns):
        print("Error: CSV missing required columns 'environment', 'steps', or 'reward_mean'")
        return

    thresholds = {}

    for env_name, group in df.groupby("environment"):
        # Sort by Step ascending
        group_sorted = group.sort_values("steps")
        steps = group_sorted["steps"].to_list()
        rewards = pd.to_numeric(group_sorted["reward_mean"], errors="coerce").dropna().to_list()

        r_star, t_run = compute_threshold_from_rewards(rewards)
        if r_star is None:
            continue

        first_reach = find_first_step_reaching_threshold(steps, rewards, t_run)
        best_reward = max(rewards)
        best_step = steps[np.argmax(rewards)] if rewards else None

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

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved thresholds to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()