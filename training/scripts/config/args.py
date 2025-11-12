"""Command-line argument parsing."""

import argparse
from scripts.config.constants import N_STEPS
from scripts.models.data_models import TrainingArgs


def parse_arguments() -> TrainingArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ML-Agents training with system monitoring. "
                    "Created by AIML6 - Maastricht University DACS Project 2-1: AIML"
    )
    parser.add_argument("--config", required=True, help="Path to the ML-Agents YAML config file")
    parser.add_argument("--run-id", required=True, help="Run ID for the training session")
    parser.add_argument("--num-steps", type=int, default=N_STEPS, help="Number of steps to monitor")
    parser.add_argument("--headless", dest="env_path", help="Path to headless build")
    parser.add_argument("--ac", action="store_true", help="Enable auto-commit")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility")
    parser.add_argument("--no-thresholds", action="store_true", help="Disable threshold analysis")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    return TrainingArgs(
        config=args.config,
        run_id=args.run_id,
        num_steps=args.num_steps,
        ac=args.ac,
        env_path=args.env_path,
        seed=args.seed,
        no_thresholds=args.no_thresholds,
        verbose=args.verbose
    )