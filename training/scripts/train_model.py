from tensorboard.backend.event_processing import event_accumulator
import psutil
import threading
import time
import argparse
import numpy as np
import glob
import yaml
import sys
import csv
import json
import subprocess
import os

# Usage
# python train_model.py --config [config.yaml] --run-id [naming_convention] (optional: --num-steps [int])
TAGS = {
    "Environment/Cumulative Reward": "Mean Policy Reward",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy",
    "Environment/NumAgents": "Number of Agents"
}
AUTO_COMMIT_BRANCH = "main"
N_STEPS = 1000 # Default window size after the first data point

KEY_MAPPING = {
        "Run ID": "run_id",
        "Environment": "environment",
        "Seed": "seed",
        "Number of Agents": "num_agents",
        "Algorithm": "algorithm",
        "Steps": "steps",
        "Batch Size": "batch_size",
        "Buffer Size": "buffer_size",
        "Learning Rate": "learning_rate",
        "Epochs": "epochs",
        "Total Time": "total_time",
        "Average CPU": "average_cpu",
        "Average RAM": "average_ram",
        "Step Interval (Running Mean)": "step_interval",
        "Mean Policy Reward": "reward_mean",
        "Mean Policy Reward (start step)": "reward_mean_step",
        "Mean Policy Loss": "p_loss_mean",
        "Mean Policy Loss (start step)": "p_loss_mean_step",
        "Mean Value Loss": "v_loss_mean",
        "Mean Value Loss (start step)": "v_loss_mean_step",
        "Mean Entropy": "entropy_mean",
        "Mean Entropy (start step)": "entropy_mean_step",
        "Threshold Value": "threshold_value",
        "Steps to Threshold": "steps_to_threshold",
        "Time to Threshold (s)": "time_to_threshold",
        "Threshold Version": "threshold_version",
        "Run Reached Threshold": "run_reached_threshold",
        "Best Reward Before Timeout": "best_reward_before_timeout",
        "Step Of Best Reward": "step_of_best_reward",
    }


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
        if label == "Number of Agents":
            continue
        else:
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

def parse_arguments_and_load_config():
    parser = argparse.ArgumentParser(description="Run ML-Agents training with system monitoring.")
    parser.add_argument("--config", required=True, help="Path to the ML-Agents YAML config file")
    parser.add_argument("--run-id", required=True, help="Run ID for the training session")
    parser.add_argument("--num-steps", type=int, default=N_STEPS, help="Number of steps to monitor")
    parser.add_argument("--headless", dest="env_path", required=False, help="Path to the build so it can train headless (no graphics)")
    parser.add_argument("--ac", action="store_true", help="Activate auto-commit")
    parser.add_argument("--seed", type=int, help="(Optional) Seed used for data replication")
    args = parser.parse_args()

    config_file = args.config
    run_id = args.run_id
    num_steps = args.num_steps
    env_path = args.env_path
    seed = args.seed
    ac = args.ac
    if ac:
        print(f"[INFO]: Auto-commit activated for run: '{run_id}'")
    else:
        print(f"[INFO]: Auto-commit disabled. Results will not be auto-committed.")

    config_data = {}
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not read config file '{config_file}': {e}")

    return config_file, run_id, num_steps, config_data, ac, env_path, seed

def run_training_process(config_file, run_id, env_path, seed):
    print(f"\n Starting ML-Agents training")
    print(f"   Config file: {config_file}")
    print(f"   Run ID:      {run_id}")
    print(f"   Seed:        {seed}")
    
    if env_path:
        print(f"   Env build:   {env_path}")
    else:
        print("   Env build:   Connected to Unity Editor")

    results = {}

    start_time = time.time()

    stop_event = threading.Event()
    process = None
    monitor_thread = None

    try:
        # --- Launch ML-Agents training ---
        cmd = ["mlagents-learn", config_file, "--run-id", run_id, "--force", "--no-graphics"]
        if env_path:
            cmd.extend(["--env", env_path])
        if seed:
            cmd.extend(["--seed", str(seed)])
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Start system monitoring thread after a process is launched
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

    return total_time, mean_cpu, mean_ram

def analyze_training_results(run_id, config_file, num_steps, config_data, total_time, mean_cpu, mean_ram):
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
        "Algorithm": f"{get_param(('behaviors', behavior_name, 'trainer_type'))}",
        "Steps": f"{get_param(('behaviors', behavior_name, 'max_steps'))}",
        "Batch Size": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'batch_size'))}",
        "Buffer Size": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'buffer_size'))}",
        "Learning Rate": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'learning_rate'))}",
        "Epochs": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'num_epoch'))}",
        "Total Time": f"{total_time:.0f}",
        "Average CPU": f"{mean_cpu:.1f}",
        "Average RAM": f"{mean_ram:.1f}",
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
            # Update value if the key exists, else add new
            combined_data[key] = display_value

    return combined_data, behavior_name, environment, total_time, log_dir

def analyze_thresholds(run_id, behavior_name, environment, total_time, log_dir, combined_data):
    thresholds_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "data", "thresholds" ,"latest_thresholds.json")
    threshold_value = "N/A"
    steps_to_threshold = "Not reached"
    time_to_threshold = "Not reached"
    try:
        with open(thresholds_path, 'r') as f:
            thresholds_file = json.load(f)
        # Determine environment name for threshold lookup
        env_name_for_threshold = behavior_name if behavior_name else environment
        thresholds = thresholds_file["thresholds"]
        if env_name_for_threshold in thresholds:
            env_threshold_entry = thresholds[environment]
            threshold_value = env_threshold_entry["T_run"] if isinstance(env_threshold_entry, dict) else env_threshold_entry
            # Load TensorBoard events for cumulative reward
            event_files = glob.glob(
                os.path.join(log_dir, run_id, "**", "events.out.tfevents.*"),
                recursive=True
            )
            if event_files:
                ea = event_accumulator.EventAccumulator(event_files[0])
                ea.Reload()
                try:
                    reward_events = ea.Scalars("Environment/Cumulative Reward")
                    if reward_events:
                        # Sort events by step
                        reward_events.sort(key=lambda el: el.step)
                        # Find the first step where mean reward >= threshold
                        threshold_reached_step = None
                        for e in reward_events:
                            if e.value >= threshold_value:
                                threshold_reached_step = e.step
                                break
                        if threshold_reached_step is not None:
                            steps_to_threshold = threshold_reached_step
                            # Estimate time to the threshold as a proportion of total steps * total time
                            # Find the first step in reward_events
                            first_step = reward_events[0].step
                            last_step = reward_events[-1].step
                            total_steps = last_step - first_step if last_step > first_step else 1
                            elapsed_ratio = (threshold_reached_step - first_step) / total_steps
                            time_to_threshold = elapsed_ratio * total_time
                            time_to_threshold = f"{time_to_threshold:.1f}"
                        else:
                            steps_to_threshold = "Not reached"
                            time_to_threshold = "Not reached"
                    else:
                        print(f"[WARNING] No reward events found in TensorBoard logs for run '{run_id}'")
                except KeyError:
                    print(f"[WARNING] 'Environment/Cumulative Reward' tag missing in TensorBoard logs for run '{run_id}'")
            else:
                print(f"[WARNING] No TensorBoard event files found for run '{run_id}' to analyze threshold")
        else:
            print(f"[WARNING] Threshold for environment '{env_name_for_threshold}' not found in '{thresholds_path}'")
    except FileNotFoundError:
        print(f"[WARNING] Threshold file '{thresholds_path}' not found.")
    except json.JSONDecodeError:
        print(f"[WARNING] Could not parse JSON in threshold file '{thresholds_path}'.")

    combined_data["Threshold Value"] = str(threshold_value)
    combined_data["Steps to Threshold"] = str(steps_to_threshold)
    combined_data["Time to Threshold (s)"] = str(time_to_threshold)
    combined_data["Threshold Version"] = str(thresholds_file["version"])

def save_and_display_results(combined_data):
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
        # Identifiers
        "run_id", "environment", "seed", "num_agents",

        # Training configuration
        "algorithm", "steps", "batch_size", "buffer_size",
        "learning_rate", "epochs",

        # System performance
        "total_time", "average_cpu", "average_ram",

        # Tensorboard metrics
        "step_interval", "reward_mean", "reward_mean_step",
        "p_loss_mean", "p_loss_mean_step", "v_loss_mean", "v_loss_mean_step",
        "entropy_mean", "entropy_mean_step",

        # Threshold analysis
        "threshold_value", "steps_to_threshold", "time_to_threshold", "threshold_version",

        # Future-fields for predictions
        "run_reached_threshold", "best_reward_before_timeout", "step_of_best_reward"
    ]


    csv_file = os.path.join(data_dir, "combined_results.csv")
    json_file = os.path.join(data_dir, "combined_results.json")

    normalized_data = {}
    for old_key, new_key in KEY_MAPPING.items():
        if old_key in combined_data:
            normalized_data[new_key] = combined_data[old_key]
        else:
            normalized_data[new_key] = ""

    # Write or append to CSV
    file_exists = os.path.isfile(csv_file)
    # Ensure the file ends with a newline before appending
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
        filtered_data = {key: normalized_data.get(key, "") for key in CSV_HEADERS}
        print("[DEBUG] Writing row keys:", filtered_data.keys())
        writer.writerow(filtered_data)
        print(f"[INFO] Results saved to '{csv_file}'")

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
        print(f"[INFO] Results saved to '{json_file}'")


def auto_commit_results(commit_message="Auto-update: new training results"):
    """Safely auto-commit updated dataset files after validation."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    files_to_commit = ["combined_results.csv", "combined_results.json"]

    try:
        # Verify branch is selected branch
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        if branch != AUTO_COMMIT_BRANCH:
            print(f"[WARNING] Current branch is '{branch}', not designated auto commit branch '{AUTO_COMMIT_BRANCH}'")
            print(f"[INFO] To change to auto-commit branch, run 'git checkout {AUTO_COMMIT_BRANCH}'")
            print(f"[WARNING] Aborting auto-commit.")
            return

        # Check for a clean working tree
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        status_lines = status.stdout.strip().splitlines()

        def _should_ignore(line: str) -> bool:
            if not line:
                return True
            # Porcelain format: two status chars + space + path
            xy = line[:2]
            raw_path = line[3:].strip()
            if " -> " in raw_path:
                raw_path = raw_path.split(" -> ", 1)[-1].strip()

            # Ignore untracked files entirely
            if xy == "??":
                return True

            # Allow dataset result files to be dirty (staged or unstaged)
            allowed_dirty = ["data/combined_results.csv", "data/combined_results.json"]
            for allowed in allowed_dirty:
                if allowed in line or allowed in raw_path:
                    return True
            print(f"[WARNING] Unrecognized porcelain line: {line}")
            return False

        dirty_changes = [ln for ln in status_lines if not _should_ignore(ln)]

        if any(ln.strip() for ln in dirty_changes):
            print(
                "[WARNING] Working directory not clean (excluding results files). Please commit or stash your changes before training.")
            print(f"[WARNING] Aborting auto-commit.")
            return
        else:
            print("[INFO] Working directory clean of other changes (excluding results files).")

        # Ensure up to date with origin
        subprocess.run(["git", "fetch", "origin"], check=True)
        local = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        remote = subprocess.check_output(["git", "rev-parse", f"origin/{AUTO_COMMIT_BRANCH}"]).decode().strip()
        if local != remote:
            print(f"[WARNING] Local branch not up to date with origin/{AUTO_COMMIT_BRANCH}. Please pull first.")
            print(f"[WARNING] Aborting auto-commit.")
            return

        # Stage files
        print(f"[INFO] Staging dataset files for commit on branch '{branch}'")
        for file in files_to_commit:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                subprocess.run(["git", "add", file_path], check=True)
            else:
                print(f"[WARNING] File not found: {file}")

        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode != 0:  # there are changes staged
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            subprocess.run(["git", "push"], check=True)
            print("[INFO] Auto-commit completed successfully.")
        else:
            print("[INFO] No dataset changes to commit.")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git command failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def main():
    config_file, run_id, num_steps, config_data, ac, env_path, seed = parse_arguments_and_load_config()
    total_time, mean_cpu, mean_ram = run_training_process(config_file, run_id, env_path, seed)
    combined_data, behavior_name, environment, total_time, log_dir = analyze_training_results(run_id, config_file, num_steps, config_data, total_time, mean_cpu, mean_ram)
    analyze_thresholds(run_id, behavior_name, environment, total_time, log_dir, combined_data)
    save_and_display_results(combined_data)
    if ac:
        auto_commit_results(f"Auto-update: new training results for {run_id}")

if __name__ == "__main__":
    main()