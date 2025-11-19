"""
ML-Agents Training Runner Version 2.5
Created by AIML 6 2024-2025 (Maastricht University - DACS P2-1: Artificial Intelligence and Machine Learning)

usage: python -m scripts.train_model [-h] [--version] [--verbose] [-v] --config CONFIG --run-id RUN_ID [--num-steps NUM_STEPS] [--headless ENV_PATH] [--ac] [--seed SEED] [--no-thresholds] [--batch]
options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --config              Path to the ML-Agents YAML config file
  --run-id              Run ID for the training session
  --num-steps           (Optional) Number of steps to monitor. Uses configured value if not provided
  --headless            (Optional) Path to the build so it can train headless (no graphics)
  --ac                  (Optional) Activate auto-commit
  --seed                (Optional) Seed used for data replication
  --no-thresholds       (Optional) Disable thresholds for this run
  --batch               (Optional) Enable batch training mode with seed range
  -v, --verbose         (Optional) Enable verbose mlagents-learn output
"""


from scripts.config.args import parse_arguments
from scripts.core.training import TrainingRunner
from scripts.utils.display import print_intro, save_and_display_results
from scripts.utils.git_utils import auto_commit_results

def main():
    args = parse_arguments()
    print_intro(args)
    if args.batch_range is not None:
        start, end = args.batch_range
        run_id = args.run_id
        for i in range(start, end + 1):
            print(f"[INFO] Starting training run {i}")
            args.seed = i
            args.run_id = f"{run_id}_{i}"
            training = TrainingRunner(args)
            result = training.run()
            save_and_display_results(result.combined_data, args.verbose)
        if args.ac:
            auto_commit_results(args.verbose,f"Auto-update: new training batch results for {run_id} seed range {start}-{end}")
    else:
        training = TrainingRunner(args)
        result = training.run()
        save_and_display_results(result.combined_data, args.verbose)
        if args.ac:
            auto_commit_results(args.verbose,f"Auto-update: new training results for {args.run_id}")

if __name__ == "__main__":
    main()