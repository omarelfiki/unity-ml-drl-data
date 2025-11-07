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
# TODO: Define paths for data files and output report.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_dir = os.path.join(project_root, "data")
csv_path = os.path.join(data_dir, "combined_results.csv")
json_path = os.path.join(data_dir, "combined_results.json")
report_md = os.path.join(data_dir, "validation_report.md")

log(f"Project root: {project_root}")
log(f"Target file: {csv_path}")

# === STEP 1: LOAD DATA ===
# TODO:
# - Load combined_results.csv into pandas DataFrame
# - Optionally cross-check JSON consistency
try:
    df = pd.read_csv(csv_path)
    df = df.fillna("N/A")
    log(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
except Exception as e:
    log(f"[ERROR] Could not load CSV: {e}")
    df = pd.DataFrame()

# === STEP 2: DEFINE VALIDATION RULES ===
# TODO:
# - Define expected columns, allowed value ranges, and required data types
expected_columns = [
    "Run ID", "Environment", "Seed", "Number of Agents", "Algorithm", "Steps",
    "Batch Size", "Buffer Size", "Learning Rate", "Epochs",
    "Total Time (s)", "Average CPU (%)", "Average RAM (%)",
    "Mean Policy Reward", "Mean Policy Loss", "Mean Value Loss", "Mean Entropy"
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

    check_range("Average CPU (%)", 0, 100)
    check_range("Average RAM (%)", 0, 100)

    # Type consistency check
    for col in ["Total Time (s)", "Mean Policy Reward"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            validation_issues.append(f"Invalid data type in column: {col}")

else:
    validation_issues.append("CSV file is empty or unreadable.")

# === STEP 3: CROSS-VERIFY JSON (Optional) ===
# TODO:
# - Optionally compare number of entries between CSV and JSON.
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
# TODO:
# - Create a Markdown summary file summarizing all checks.
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