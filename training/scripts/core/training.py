"""Core training execution and orchestration."""

import subprocess
import yaml

from scripts.config.args import TrainingArgs
from scripts.core.metrics import MetricsAnalyzer
from scripts.core.system_monitor import SystemMonitor
from scripts.models.data_models import TrainingResult


class TrainingRunner:
    def __init__(self, args: TrainingArgs):
        self.args = args
        self.config_data = self._load_config()
        self.metrics_analyzer = MetricsAnalyzer(args)

    def _load_config(self) -> dict:
        """Load YAML configuration file."""
        try:
            with open(self.args.config, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Could not read config '{self.args.config}': {e}")
            return {}

    def run(self) -> TrainingResult:
        """Execute complete training pipeline."""
        performance = self._run_training()
        result = self.metrics_analyzer.analyze(self.config_data, performance)

        if not self.args.no_thresholds:
            self.metrics_analyzer.analyze_thresholds(result)

        return result

    def _run_training(self):
        """Execute ML-Agents training with system monitoring."""
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
        """Build mlagents-learn command."""
        cmd = [
            "mlagents-learn",
            self.args.config,
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
        """Stream and filter ML-Agents output."""
        for line in iter(process.stdout.readline, ""):
            if self.args.verbose or line.startswith("[INFO]"):
                print(line, end="")