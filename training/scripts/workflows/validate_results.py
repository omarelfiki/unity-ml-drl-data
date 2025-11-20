"""
Generates a Markdown validation report for training runs.

Reads data/combined_results.csv, and validates data on values, ranges and types.

Dependency for Github Actions workflows on repo.
"""
import sys

# === Imports ===
import pandas as pd
import json
import os
import time
from datetime import datetime


# === SIMPLE LOGGING ===
def log(message):
    """Timestamped log messages for clarity in local runs and GitHub Actions."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def exit_message():
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    log(f"Finished in {duration} seconds.")
    log("=== Validation Process Complete ===")

# Start
log("=== Starting Validation Process ===")
start_time = time.time()

# === PATH SETUP ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
data_dir = os.path.join(project_root, "data")
csv_path = os.path.join(data_dir, "combined_results.csv")
json_path = os.path.join(data_dir, "combined_results.json")
report_md = os.path.join(data_dir, "validation_report.md")

log(f"Project root: {project_root}")
log(f"Target file: {csv_path}")

# === STEP 1: LOAD DATA ===
try:
    df = pd.read_csv(csv_path)
    df = df.fillna("N/A")
    log(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
except Exception as e:
    log(f"[ERROR] Could not load CSV: {e}")
    df = pd.DataFrame()

# === STEP 2: DEFINE VALIDATION RULES ===
expected_columns = [
    # Identifiers
    "run_id", "environment", "seed", "num_agents",
    "algorithm", "steps", "batch_size", "buffer_size",
    "learning_rate", "epochs",
    # System Performance Metrics
    "total_time", "average_cpu", "average_ram",
    # Training Metrics
    "reward_mean", "reward_mean_step", "early_reward_mean", "early_reward_mean_step",
    "final_reward_mean", "final_reward_mean_step", "best_reward", "best_reward_step",
    "step_interval", "p_loss_mean", "p_loss_mean_step", "v_loss_mean",
    "v_loss_mean_step", "entropy_mean", "entropy_mean_step",
    # Threshold Analysis
    "threshold_value", "steps_to_threshold", "time_to_threshold",
    "threshold_version", "run_reached_threshold",
    # misc
    "notes"
]

validation_issues = []

if not df.empty:
    # Missing column check
    for col in expected_columns:
        if col not in df.columns:
            validation_issues.append(f"Missing column: {col}")

    # Value range check
    def check_range(col, min_val, max_val):
        if col in df.columns:
            invalid = df[(df[col].astype(float) < min_val) | (df[col].astype(float) > max_val)]
            if not invalid.empty:
                validation_issues.append(f"Out-of-range values in '{col}' ({len(invalid)} rows)")

    def check_duplicates(df, col="run_id"):
        if col in df.columns:
            dup_count = df[col].duplicated().sum()
            if dup_count:
                validation_issues.append(f"Duplicate values in '{col}': {dup_count}")

    def check_seed_integer(df, col="seed"):
        if col in df.columns:
            # Convert to numeric, produce NaN for invalid values
            coerced = pd.to_numeric(df[col], errors="coerce")

            # Identify rows where seed is invalid (NaN) but original value is not N/A
            invalid_mask = coerced.isna() & (df[col] != "N/A")

            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                validation_issues.append(f"Non-integer values in '{col}': {invalid_count}")

    def check_threshold_versions(col):
        """Ensure that all threshold versions referenced in CSV exist in data/thresholds/."""
        if col in df.columns:
            thresholds_dir = os.path.join(data_dir, "thresholds")
            if not os.path.exists(thresholds_dir):
                validation_issues.append("Thresholds directory missing.")
                return
            existing_files = {f.replace("thresholds_", "").replace(".json", "")
                              for f in os.listdir(thresholds_dir)
                              if f.startswith("thresholds_") and f.endswith(".json")}

            missing_versions = set(df[col].unique()) - existing_files - {"N/A"}
            if missing_versions:
                validation_issues.append(
                    f"Missing threshold version files for: {', '.join(missing_versions)}"
                )

    check_range("average_cpu", 0, 100)
    check_range("average_ram", 0, 100)
    check_seed_integer(df)
    check_duplicates(df)
    check_threshold_versions("threshold_version")

    # Type consistency check
    for col in ["total_time", "reward_mean"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            validation_issues.append(f"Invalid data type in column: {col}")

else:
    validation_issues.append("CSV file is empty or unreadable.")

try:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        if len(json_data) != len(df):
            validation_issues.append(
                f"Mismatch: JSON has {len(json_data)} entries, CSV has {len(df)}."
            )
        log("JSON consistency check complete.")
except Exception as e:
    log(f"[WARNING] Could not verify JSON file: {e}")

# === STEP 4: GENERATE VALIDATION REPORT ===
try:
    with open(report_md, "w") as f:
        f.write("# Data Validation Report\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if not validation_issues:
            f.write("**All checks passed successfully.**\n")
        else:
            f.write("⚠️ **Issues found:**\n")
            for issue in validation_issues:
                f.write(f"- {issue}\n")

        f.write("\n## Dataset Overview\n")
        if not df.empty:
            f.write(f"- Rows: {len(df)}\n")
            f.write(f"- Columns: {len(df.columns)}\n")
            f.write(f"- Missing values: {df.isna().sum().sum()}\n")

    log(f"Validation report written to: {report_md}")

except Exception as e:
    log(f"[ERROR] Failed to write validation report: {e}")

# === STEP 5: ACTION OUTCOME ===
if validation_issues:
    log("Validation failed with issues:")
    for issue in validation_issues:
        log(f"- {issue}")
    exit_message()
    sys.exit(1)
else:
    log("Validation completed with no issues.")
    exit_message()
    sys.exit(0)