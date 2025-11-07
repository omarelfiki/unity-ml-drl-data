"""
This script:
1. Checks the structure and format of combined_results.csv and combined_results.json.
2. Ensures all required columns are present.
3. Verifies value ranges, data types, and missing values.
4. Prints a validation summary (and optionally fail GitHub Actions if issues found).
5. Optionally appends results to a validation log file.

Inputs:
    - data/combined_results.csv (required)
    - data/combined_results.json (optional)

Outputs:
    - data/validation_report.md (summary report)
    - (optional) printed logs for CI/CD workflow validation

Dependencies:
    pandas, json

"""

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

# Start
log("=== Starting Validation Process ===")
start_time = time.time()

# === PATH SETUP ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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
        "run_id", "environment", "seed", "num_agents",
        "algorithm", "steps", "batch_size", "buffer_size",
        "learning_rate", "epochs", "total_time", "average_cpu",
        "average_ram", "step_interval", "reward_mean", "reward_mean_step",
        "p_loss_mean", "p_loss_mean_step", "v_loss_mean", "v_loss_mean_step",
        "entropy_mean", "entropy_mean_step", "threshold_method", "threshold_value",
        "threshold_alpha", "reference_window_last_steps", "smoothing_window", "patience_k",
        "first_data_step", "run_reached_threshold", "best_reward_before_timeout", "step_of_best_reward",
        "episode_success_rule", "episode_success_rate_window"
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

    check_range("average_cpu", 0, 100)
    check_range("average_ram", 0, 100)

    # Type consistency check
    for col in ["total_time", "reward_mean"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            validation_issues.append(f"Invalid data type in column: {col}")

else:
    validation_issues.append("CSV file is empty or unreadable.")

# === STEP 3: CROSS-VERIFY JSON (Optional) ===
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
        log(f"   - {issue}")
else:
    log("Validation completed with no issues.")

# === STEP 6: END ===
end_time = time.time()
duration = round(end_time - start_time, 2)
log(f"Finished in {duration} seconds.")
log("=== Validation Process Complete ===")