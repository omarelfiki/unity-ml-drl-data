from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import glob
import os
import sys

# --- Default parameters ---
DEFAULT_LOG_DIR = "results"
N_STEPS = 1000  # window size after first data point

TAGS = {
    "Environment/Cumulative Reward": "Mean Policy Reward",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy"
}


def extract_metrics(run_id, log_dir):
    """Extract mean metrics over the first N_STEPS after the first logged step."""
    # Locate TensorBoard event file
    event_files = glob.glob(
        os.path.join(log_dir, run_id, "**", "events.out.tfevents.*"),
        recursive=True
    )
    if not event_files:
        print(f"Warning: No TensorBoard logs found for run '{run_id}'")
        return None

    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    metrics = {"Run ID": run_id}

    for tag, label in TAGS.items():
        try:
            events = ea.Scalars(tag)
        except KeyError:
            print(f"Warning: Missing tag '{tag}' in '{run_id}'")
            metrics[label] = None
            continue

        if not events:
            metrics[label] = None
            continue

        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        # Determine first available step and define window
        first_step = steps.min()
        window_limit = first_step + N_STEPS
        mask = (steps >= first_step) & (steps <= window_limit)

        if np.any(mask):
            metrics[label] = float(values[mask].mean())
            metrics[f"{label} (start step)"] = int(first_step)
        else:
            metrics[label] = None
            metrics[f"{label} (start step)"] = int(first_step)

    return metrics


def main():
    # Parse results directory from arguments
    log_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_DIR
    if not os.path.isdir(log_dir):
        print(f"Error: Invalid directory '{log_dir}'")
        sys.exit(1)

    print(f"\nAnalyzing runs under '{log_dir}' — averaging first {N_STEPS} steps after data starts\n")

    found = False
    for run_id in sorted(os.listdir(log_dir)):
        run_path = os.path.join(log_dir, run_id)
        if not os.path.isdir(run_path):
            continue

        found = True
        result = extract_metrics(run_id, log_dir)
        if not result:
            continue

        print(f"\nRun: {result['Run ID']}")
        print("-" * 60)
        for key, value in result.items():
            if key == "Run ID":
                continue
            if "start step" in key:
                print(f"{key:<35}: {value}")
            elif value is None:
                print(f"{key:<35}: (no data)")
            else:
                print(f"{key:<35}: {value:.4f}")
        print("-" * 60)

    if not found:
        print("No valid runs found in this directory.")


if __name__ == "__main__":
    main()