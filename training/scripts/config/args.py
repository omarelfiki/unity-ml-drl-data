"""Command-line argument parsing."""

import argparse
import os

from scripts.config.constants import DEFAULT_N_STEPS
from scripts.models.data_models import TrainingArgs


def parse_arguments() -> TrainingArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ML-Agents training with system monitoring. "
                    "Created by AIML6 - Maastricht University DACS Project 2-1: AIML"
    )
    parser.add_argument("--config", required=True, help="Path to the ML-Agents YAML config file")
    parser.add_argument("--run-id", required=True, help="Run ID for the training session")
    parser.add_argument("--num-steps", type=int, help="Number of steps to monitor")
    parser.add_argument("--headless", dest="env_path", help="Path to headless build")
    parser.add_argument("--ac", action="store_true", help="Enable auto-commit")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility")
    parser.add_argument("--no-thresholds", action="store_true", help="Disable threshold analysis")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", action="version", version="%(prog)s 2.0")
    parser.add_argument("--batch", nargs=2, type = int, help="Set a range of seeds for batch training")
    parser.add_argument("--randomize", action="store_true", help="Randomize the config file for selected environment when using --batch")

    args = parser.parse_args()
    batch_range = tuple(args.batch) if args.batch else None
    if args.seed is not None and batch_range is not None:
        raise ValueError("Cannot specify both --seed and --batch")
    if args.randomize:
        if batch_range is None:
            raise ValueError("Cannot randomize config without batch range.")

    return TrainingArgs(
        config=args.config,
        run_id=args.run_id,
        num_steps=args.num_steps,
        ac=args.ac,
        env_path=args.env_path,
        seed=args.seed,
        no_thresholds=args.no_thresholds,
        verbose=args.verbose,
        batch_range=batch_range,
        randomize=args.randomize
    )