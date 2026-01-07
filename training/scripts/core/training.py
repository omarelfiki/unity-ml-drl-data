"""Core training execution and orchestration."""

import subprocess
import yaml
import tempfile
import os
import random

from scripts.config.args import TrainingArgs
from scripts.core.metrics import MetricsAnalyzer
from scripts.core.system_monitor import SystemMonitor
from scripts.models.data_models import TrainingResult


class TrainingRunner:
    def __init__(self, args: TrainingArgs):
        self.args = args
        self.config_data = self._load_config()
        self.metrics_analyzer = MetricsAnalyzer(args)
        self._temp_config_path = None

    def _load_config(self) -> dict:
        try:
            with open(self.args.config, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Could not read config '{self.args.config}': {e}")
            return {}

    # Create randomized config and write to temp file
    def _build_randomized_config(self) -> str:
        cfg = dict(self.config_data) if self.config_data is not None else {}
        behaviors = cfg.get("behaviors", {})

        if not behaviors:
            print("[WARNING] No behaviors found in config; skipping randomization.")
            return self.args.config

        behavior_name = next(iter(behaviors))
        behavior_cfg = behaviors[behavior_name]

        # Algorithm: 50% PPO, 50% SAC
        new_algo = random.choice(["ppo", "sac"])
        behavior_cfg["trainer_type"] = new_algo

        # Learning rate
        hyper = behavior_cfg.get("hyperparameters", {})
        lr = hyper.get("learning_rate")
        if isinstance(lr, (int, float)):
            scale = random.uniform(0.1, 3)
            hyper["learning_rate"] = lr * scale
            print(f"[INFO] Original learning_rate: {lr}, scale: {scale:.3f}, new: {hyper['learning_rate']}")
        else:
            print("[WARNING] learning_rate missing or non-numeric; skipping LR randomization.")

        # Batch + buffer
        factor = random.uniform(0.1, 3.0)
        bs = hyper.get("batch_size")
        buf = hyper.get("buffer_size")

        if isinstance(bs, int):
            new_bs = max(1, int(bs * factor))
            hyper["batch_size"] = new_bs
            print(f"[INFO] Original batch_size: {bs}, factor: {factor}, new: {hyper['batch_size']}")
        else:
            print("[WARNING] batch_size missing or non-int; skipping batch randomization.")

        if isinstance(buf, int):
            new_buf = max(1, int(buf * factor))
            hyper["buffer_size"] = new_buf
            print(f"[INFO] Original buffer_size: {buf}, factor: {factor}, new: {hyper['buffer_size']}")
        else:
            print("[WARNING] buffer_size missing or non-int; skipping buffer randomization.")

        behavior_cfg["hyperparameters"] = hyper
        behaviors[behavior_name] = behavior_cfg
        cfg["behaviors"] = behaviors

        # Write to temp YAML
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
        try:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            self._temp_config_path = tmp.name
        finally:
            tmp.close()

        return self._temp_config_path

    def run(self) -> TrainingResult:
        performance = self._run_training()
        result = self.metrics_analyzer.analyze(self.config_data, performance)

        if not self.args.no_thresholds:
            self.metrics_analyzer.analyze_thresholds(result)

        # cleanup temp config if created
        if self._temp_config_path and os.path.exists(self._temp_config_path):
            os.remove(self._temp_config_path)

        return result

    def _run_training(self):
        if self.args.verbose:
            print("\n[INFO] Starting ML-Agents training")

        monitor = SystemMonitor()
        process = None

        try:
            cmd = self._build_command()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            monitor.start()
            self._stream_output(process)
            process.wait()

        finally:
            performance = monitor.stop()
            if process and process.poll() is None:
                process.terminate()

        return performance

    def _build_command(self) -> list[str]:
        if self.args.randomize:
            # use randomized temp config instead of original
            cfg_path = self._build_randomized_config()
        else:
            cfg_path = self.args.config

        cmd = [
            "mlagents-learn",
            cfg_path,
            "--run-id", self.args.run_id,
            "--force",
            "--no-graphics"
        ]

        if self.args.env_path:
            cmd.extend(["--env", self.args.env_path])
        if self.args.seed:
            cmd.extend(["--seed", str(self.args.seed)])

        return cmd

    def _stream_output(self, process):
        for line in iter(process.stdout.readline, ""):
            if self.args.verbose or line.startswith("[INFO]"):
                print(line, end="")
