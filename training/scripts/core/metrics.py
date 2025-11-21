import glob
import json
import os

import numpy as np
import yaml
from tensorboard.backend.event_processing import event_accumulator
from scripts.models.data_models import TrainingResult, TrainingArgs
from scripts.config.constants import get_n_steps_for_env, TAGS, get_early_window_for_env, get_final_window_for_env


class MetricsAnalyzer:
    def __init__(self, args: TrainingArgs):
        self.args = args

    def analyze(self, config_data, performance):
        v = self.args.verbose
        log_dir = os.path.join(os.path.dirname(__file__),".." ,"..", "results")
        log_dir = os.path.abspath(log_dir)
        env_name = os.path.splitext(os.path.basename(self.args.config))[0]
        if self.args.num_steps is not None:
            num_steps = self.args.num_steps
        else:
            num_steps = get_n_steps_for_env(env_name)
        metrics = self.extract_tensorboard_metrics(log_dir, num_steps)
        environment = os.path.splitext(os.path.basename(self.args.config))[0]

        # Load generated configuration files after training
        generated_config = {}
        generated_config_path = os.path.join(log_dir, self.args.run_id, "configuration.yaml")
        if os.path.exists(generated_config_path):
            try:
                with open(generated_config_path, "r") as f:
                    generated_config = yaml.safe_load(f)
                    if v: print("[INFO] Loaded training-generated config file:")
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
            "Run ID": self.args.run_id,
            "Environment": behavior_name,
            "Seed": str(seed),
            "Algorithm": f"{get_param(('behaviors', behavior_name, 'trainer_type'))}",
            "Steps": f"{get_param(('behaviors', behavior_name, 'max_steps'))}",
            "Batch Size": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'batch_size'))}",
            "Buffer Size": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'buffer_size'))}",
            "Learning Rate": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'learning_rate'))}",
            "Epochs": f"{get_param(('behaviors', behavior_name, 'hyperparameters', 'num_epoch'))}",
            "Total Time": f"{performance.total_time:.0f}",
            "Average CPU": f"{performance.mean_cpu:.1f}",
            "Average RAM": f"{performance.mean_ram:.1f}",
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

        return TrainingResult(combined_data, behavior_name, environment, performance.total_time, log_dir)

    def analyze_thresholds(self, result: TrainingResult):
        thresholds_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")), "data",
                                       "thresholds", "latest_thresholds.json")
        threshold_value = "N/A"
        steps_to_threshold = "Not reached"
        time_to_threshold = "Not reached"
        threshold_reached = False

        try:
            with open(thresholds_path, 'r') as f:
                thresholds_file = json.load(f)
            # Determine environment name for threshold lookup
            env_name_for_threshold = result.behavior_name if result.behavior_name else result.environment
            thresholds = thresholds_file["thresholds"]
            if env_name_for_threshold in thresholds:
                env_threshold_entry = thresholds[result.environment]
                threshold_value = env_threshold_entry["T_run"] if isinstance(env_threshold_entry,
                                                                             dict) else env_threshold_entry
                # Load TensorBoard events for cumulative reward
                event_files = glob.glob(
                    os.path.join(result.log_dir, self.args.run_id, "**", "events.out.tfevents.*"),
                    recursive=True
                )
                if event_files:
                    ea = event_accumulator.EventAccumulator(event_files[0])
                    ea.Reload()
                    try:
                        reward_events = ea.Scalars("Policy/Extrinsic Reward")
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
                                time_to_threshold = elapsed_ratio * result.total_time
                                time_to_threshold = f"{time_to_threshold:.1f}"
                                threshold_reached = True
                            else:
                                steps_to_threshold = "Not reached"
                                time_to_threshold = "Not reached"
                        else:
                            print(f"[WARNING] No reward events found in TensorBoard logs for run '{self.args.run_id}'")
                    except KeyError:
                        print(
                            f"[WARNING] 'Environment/Cumulative Reward' tag missing in TensorBoard logs for run '{self.args.run_id}'")
                else:
                    print(f"[WARNING] No TensorBoard event files found for run '{self.args.run_id}' to analyze threshold")
            else:
                print(
                    f"[WARNING] Threshold for environment '{env_name_for_threshold}' not found in '{thresholds_path}'")
        except FileNotFoundError:
            print(f"[WARNING] Threshold file '{thresholds_path}' not found.")
        except json.JSONDecodeError:
            print(f"[WARNING] Could not parse JSON in threshold file '{thresholds_path}'.")

        result.combined_data["Threshold Value"] = str(threshold_value)
        result.combined_data["Steps to Threshold"] = str(steps_to_threshold)
        result.combined_data["Time to Threshold (s)"] = str(time_to_threshold)
        result.combined_data["Threshold Version"] = str(thresholds_file["version"])
        result.combined_data["Threshold Reached"] = str(threshold_reached)

    def extract_tensorboard_metrics(self, log_dir, n_steps):
        run_id = self.args.run_id
        environment = os.path.splitext(os.path.basename(self.args.config))[0]
        event_files = glob.glob(
            os.path.join(log_dir, run_id, "**", "events.out.tfevents.*"),
            recursive=True,
        )

        if not event_files:
            print(f"[WARNING] No TensorBoard logs found for run '{run_id}'")
            return {"Run ID": run_id}

        ea = event_accumulator.EventAccumulator(event_files[0])
        ea.Reload()

        metrics = {"Run ID": run_id}
        try:
            reward_events = ea.Scalars("Environment/Cumulative Reward")
        except KeyError:
            reward_events = []

        if reward_events:
            reward_events.sort(key=lambda e: e.step)

            steps = np.array([e.step for e in reward_events])
            values = np.array([e.value for e in reward_events])

            first = steps.min()
            last = steps.max()

            # Window for reward_mean
            mask = (steps >= first) & (steps <= first + n_steps)
            metrics["Mean Policy Reward"] = float(values[mask].mean()) if np.any(mask) else None
            metrics["Mean Policy Reward (start step)"] = int(first)

            # Early window
            early_window = get_early_window_for_env(environment)
            early_mask = (steps >= first) & (steps <= first + early_window)
            metrics["Early Reward Mean"] = float(values[early_mask].mean()) if np.any(early_mask) else None
            metrics["Early Reward Mean (start step)"] = int(first)

            # Final window
            final_window = get_final_window_for_env(environment)
            final_mask = (steps >= max(first, last - final_window)) & (steps <= last)
            metrics["Final Reward Mean"] = float(values[final_mask].mean()) if np.any(final_mask) else None
            metrics["Final Reward Mean (start step)"] = int(max(first, last - final_window))

            # Best reward
            idx = np.argmax(values)
            metrics["Best Reward"] = float(values[idx])
            metrics["Best Reward (step)"] = int(steps[idx])

        else:
            metrics["Mean Policy Reward"] = None
            metrics["Mean Policy Reward (start step)"] = None
            metrics["Early Reward Mean"] = None
            metrics["Early Reward Mean (start step)"] = None
            metrics["Final Reward Mean"] = None
            metrics["Final Reward Mean (start step)"] = None
            metrics["Best Reward"] = None
            metrics["Best Reward (step)"] = None

        for tag, label in TAGS.items():
            try:
                events = ea.Scalars(tag)
            except KeyError:
                metrics[label] = None
                metrics[f"{label} (start step)"] = None
                continue

            if not events:
                metrics[label] = None
                metrics[f"{label} (start step)"] = None
                continue

            events.sort(key=lambda e: e.step)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])

            first = steps.min()
            mask = (steps >= first) & (steps <= first + n_steps)

            metrics[label] = float(values[mask].mean()) if np.any(mask) else None
            if label != "Number of Agents":
                metrics[f"{label} (start step)"] = int(first)

        all_steps = []
        for tag in TAGS.keys():
            try:
                events = ea.Scalars(tag)
                all_steps.extend([e.step for e in events])
            except Exception:
                pass

        metrics["Step Interval (Running Mean)"] = (
            int(max(all_steps) - min(all_steps)) if all_steps else None
        )

        return metrics