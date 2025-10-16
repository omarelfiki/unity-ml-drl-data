from tensorboard.backend.event_processing import event_accumulator
import psutil
import subprocess
import threading
import time
import argparse
import numpy as np
import glob
import os
import yaml
import sys
import csv
import json

# Usage
# python train_model.py --config [config.yaml] --run-id [naming_convention] (optional: --num-steps [int])

TAGS = {
    "Environment/Cumulative Reward": "Mean Policy Reward",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy"
}

N_STEPS = 1000 # Default window size after first data point

def extract_metrics(run_id, log_dir, n_steps):
    event_files = glob.glob(
        os.path.join(log_dir, run_id, "**", "events.out.tfevents.*"),
        recursive=True
    )
    if not event_files:
        print(f"[WARNING] No TensorBoard logs found for run '{run_id}'")
        # Return all metrics with None to ensure table consistency
        return {label: None for label in TAGS.values()} | {
            f"{label} (start step)": None for label in TAGS.values()
        } | {"Run ID": run_id}

    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    metrics = {"Run ID": run_id}
    for tag, label in TAGS.items():
        try:
            events = ea.Scalars(tag)
        except KeyError:
            print(f"[WARNING] Missing tag '{tag}' in '{run_id}'")
            metrics[label] = None
            metrics[f"{label} (start step)"] = None
            continue

        if not events:
            metrics[label] = None
            metrics[f"{label} (start step)"] = None
            continue

        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        first_step = steps.min()
        window_limit = first_step + n_steps
        mask = (steps >= first_step) & (steps <= window_limit)

        metrics[label] = float(values[mask].mean()) if np.any(mask) else None
        metrics[f"{label} (start step)"] = int(first_step)

    # Compute step interval used for computing running means
    all_steps = []
    for tag in TAGS.keys():
        try:
            events = ea.Scalars(tag)
            if events:
                steps = np.array([e.step for e in events])
                all_steps.extend(steps)
        except Exception:
            continue

    if all_steps:
        first_step = min(all_steps)
        last_step = max(all_steps)
        metrics["Step Interval (Running Mean)"] = int(last_step - first_step)
    else:
        metrics["Step Interval (Running Mean)"] = None

    return metrics

def monitor_system(stop_event, interval=1):
    """Monitor CPU and RAM usage while training runs."""
    cpu_readings = []
    ram_readings = []
    start_time = time.time()

    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=interval)
        ram = psutil.virtual_memory()
        cpu_readings.append(cpu_percent)
        ram_readings.append(ram.percent)

    total_time = time.time() - start_time
    mean_cpu = sum(cpu_readings) / len(cpu_readings)
    mean_ram = sum(ram_readings) / len(ram_readings)
    return mean_cpu, mean_ram, total_time

def main():
    parser = argparse.ArgumentParser(description="Run ML-Agents training with system monitoring.")
    parser.add_argument("--config", required=True, help="Path to the ML-Agents YAML config file")
    parser.add_argument("--run-id", required=True, help="Run ID for the training session")
    parser.add_argument("--num-steps", type=int, default=N_STEPS, help="Number of steps to monitor")
    args = parser.parse_args()

    config_file = args.config
    run_id = args.run_id
    num_steps = args.num_steps

    config_data = {}
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not read config file '{config_file}': {e}")

    print(f"\n Starting ML-Agents training")
    print(f"   Config file: {config_file}")
    print(f"   Run ID:      {run_id}")

    # Define container for results from monitor thread
    results = {}

    start_time = time.time()

    stop_event = threading.Event()
    process = None
    monitor_thread = None

    try:
        # --- Launch ML-Agents training ---
        process = subprocess.Popen(
            ["mlagents-learn", config_file, "--run-id", run_id, "--force"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Start system monitoring thread after process is launched
        def monitor_wrapper():
            results["metrics"] = monitor_system(stop_event)
        monitor_thread = threading.Thread(target=monitor_wrapper)
        monitor_thread.start()

        # Stream ML-Agents output
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
        process.wait()

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Shutting down gracefully...")
        stop_event.set()
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if monitor_thread:
            monitor_thread.join()
        print("[INFO] Shutdown complete.")
        sys.exit(0)

    stop_event.set()
    if monitor_thread:
        monitor_thread.join()

    total_time = time.time() - start_time
    mean_cpu, mean_ram, _ = results["metrics"]

    # Analyze training results from tensorboard logs
    log_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    log_dir = os.path.abspath(log_dir)
    metrics = extract_metrics(run_id, log_dir, num_steps)
    environment = os.path.splitext(os.path.basename(config_file))[0]

    # Load generated configuration files after training
    generated_config = {}
    generated_config_path = os.path.join(log_dir, run_id, "configuration.yaml")
    if os.path.exists(generated_config_path):
        try:
            with open(generated_config_path, "r") as f:
                generated_config = yaml.safe_load(f)
                print("[INFO] Loaded generated config file:")
        except Exception as e:
            print(f"[WARNING] Could not read generated config file '{generated_config_path}': {e}")

    behavior_name = next(iter(generated_config.get("behaviors", {})), None)

    # Helper to get parameter, prioritizing generated_config over config_data
    def get_param(param, default="N/A"):
        # param: tuple of keys to traverse, e.g. ('behaviors', environment, 'trainer_type')
        d = generated_config
        for k in param:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                d = None
                break
        if d is not None:
            return d
        # fallback to original config files if not found in generated_config
        d = config_data
        for k in param:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    # Get seed value from generated_config if available
    seed = get_param(('env_settings', 'seed'), "N/A")

    combined_data = {
        "Run ID": run_id,
        "Environment": behavior_name,
        "Seed": str(seed),
        "Number of Agents": "Find on Unity",
        "Algorithm": f"{get_param(('behaviors', behavior_name, 'trainer_type'))}",
        "Steps": f"{get_param(('behaviors', behavior_name, 'max_steps'))}",
        "Batch Size": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'batch_size'))}",
        "Buffer Size": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'buffer_size'))}",
        "Learning Rate": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'learning_rate'))}",
        "Epochs": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'num_epoch'))}",
        "Total Time (s)": f"{total_time:.0f}",
        "Average CPU (%)": f"{mean_cpu:.1f}",
        "Average RAM (%)": f"{mean_ram:.1f}",
    }
    if metrics:
        for key, value in metrics.items():
            if key == "Run ID":
                continue
            if value is None:
                display_value = "(no data)"
            elif isinstance(value, int):
                display_value = str(value)
            elif isinstance(value, float):
                display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            # Update value if key exists, else add new
            combined_data[key] = display_value

    key_width = max(len(k) for k in combined_data.keys())
    val_width = max(len(v) for v in combined_data.values())

    print("\n" + "=" * (key_width + val_width + 7))
    print(f"| {'Metric'.ljust(key_width)} | {'Value'.ljust(val_width)} |")
    print("=" * (key_width + val_width + 7))

    for key, value in combined_data.items():
        print(f"| {key.ljust(key_width)} | {value.ljust(val_width)} |")

    print("=" * (key_width + val_width + 7))

    # Save results to CSV and JSON in 'data' folder outside 'training' directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    CSV_HEADERS = [
        "Run ID", "Environment", "Seed", "Number of Agents", "Algorithm",
        "Steps", "Batch Size", "Buffer Size", "Learning Rate", "Epochs",
        "Total Time (s)", "Average CPU (%)", "Average RAM (%)", "Step Interval (Running Mean)",
        "Mean Policy Reward", "Mean Policy Reward (start step)",
        "Mean Policy Loss", "Mean Policy Loss (start step)",
        "Mean Value Loss", "Mean Value Loss (start step)",
        "Mean Entropy", "Mean Entropy (start step)"
    ]

    csv_file = os.path.join(data_dir, "combined_results.csv")
    json_file = os.path.join(data_dir, "combined_results.json")

    # Write or append to CSV
    file_exists = os.path.isfile(csv_file)
    # Ensure file ends with a newline before appending
    with open(csv_file, 'a+', newline='') as csvfile:
        csvfile.seek(0, os.SEEK_END)
        if csvfile.tell() > 0:
            csvfile.seek(csvfile.tell() - 1)
            last_char = csvfile.read(1)
            if last_char != '\n':
                csvfile.write('\n')

        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        # Filter and enforce correct order
        filtered_data = {key: combined_data.get(key, "") for key in CSV_HEADERS}
        writer.writerow(filtered_data)

    # Write or append to JSON
    json_data = []
    if os.path.isfile(json_file):
        try:
            with open(json_file, 'r') as jf:
                json_data = json.load(jf)
        except Exception:
            json_data = []

    json_data.append(combined_data)
    with open(json_file, 'w') as jf:
        json.dump(json_data, jf, indent=4)

if __name__ == "__main__":
    main()