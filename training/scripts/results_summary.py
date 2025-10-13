"""
TODO: Generate a Markdown summary report for all collected training runs.

This script should:
1. Read the combined training results (CSV) from /data.
2. Compute overall statistics (e.g., average CPU, reward, etc.).
3. Generate inline scatter plots to visualize trends.
4. Export a Markdown report summarizing key insights.
5. (Optional) Run automatically in GitHub Actions after data updates.

Inputs:
    - data/combined_results.csv (required)
    - data/combined_results.json (optional)

Outputs:
    - data/summary_report.md

Dependencies:
    pandas, matplotlib, tabulate


DO NOT CHANGE FILE NAME OR LOCATION. Dependency for Github Actions.
"""

# === Imports ===
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import os
import time
from datetime import datetime

# === SIMPLE LOGGING ===
# Utility for consistent timestamped messages
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Start log
log("=== Starting Summary Generation ===")

# === PATH SETUP ===
# TODO: Define project root and file paths (combined_results.csv, summary_report.md)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_dir = os.path.join(project_root, "data")
csv_path = os.path.join(data_dir, "combined_results.csv")
summary_md = os.path.join(data_dir, "summary_report.md")

log(f"Project root: {project_root}")
log(f"Loading data from: {csv_path}")

# === STEP 1: LOAD DATA ===
# TODO:
# - Read 'combined_results.csv' into a pandas DataFrame.
# - Replace missing values (NaN) with 'N/A'.
# - Safely convert numeric columns like Total Time, CPU, RAM, Reward to float.

try:
    df = pd.read_csv(csv_path)
    df = df.fillna("N/A")
    numeric_cols = ["Total Time (s)", "Average CPU (%)", "Average RAM (%)", "Mean Policy Reward"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    log("✅ Loaded and cleaned data successfully.")
except Exception as e:
    log(f"[ERROR] Failed to read CSV: {e}")
    df = pd.DataFrame()

# === STEP 2: COMPUTE SUMMARY STATISTICS ===
# TODO:
# - Compute total runs, unique environments, and averages for numeric metrics.
summary = {}
if not df.empty:
    summary = {
        "Total Runs": len(df),
        "Unique Environments": df["Environment"].nunique() if "Environment" in df.columns else "N/A",
    }

    for col in numeric_cols:
        if col in df.columns:
            summary[f"Average {col}"] = round(df[col].mean(skipna=True), 2)

    log("Computed summary statistics:")
    for k, v in summary.items():
        log(f"   - {k}: {v}")
else:
    log("[WARNING] No data available for summary computation.")

# === STEP 3: GENERATE PLOTS ===
# TODO:
# - Define a helper function `create_chart(x, y, title)`:
#     • Create scatter plot
#     • Encode as Base64 for Markdown embedding

def create_chart(x, y, title):
    """Create scatter plot and return Markdown image string."""
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x], df[y], color="#1f77b4", alpha=0.7)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, linestyle="--", alpha=0.6)

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150)
    plt.close()
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"![{title}](data:image/png;base64,{img_base64})"

plots = []
# Example usage:
if all(c in df.columns for c in ["Total Time (s)", "Mean Policy Reward"]):
    plots.append(create_chart("Total Time (s)", "Mean Policy Reward", "Reward vs Training Time"))
if all(c in df.columns for c in ["Average CPU (%)", "Mean Policy Reward"]):
    plots.append(create_chart("Average CPU (%)", "Mean Policy Reward", "CPU Usage vs Reward"))
if all(c in df.columns for c in ["Average RAM (%)", "Mean Policy Reward"]):
    plots.append(create_chart("Average RAM (%)", "Mean Policy Reward", "RAM Usage vs Reward"))

log(f"Generated {len(plots)} plots.")

# === STEP 4: WRITE MARKDOWN REPORT ===
# TODO:
# - Create Markdown report with:
#     • Overview statistics
#     • Recent runs (last 5)
#     • Embedded plots

try:
    with open(summary_md, "w") as f:
        f.write("# Training Summary Report\n\n")
        f.write("Automatically generated after each new data update.\n\n")

        f.write("## Overview\n")
        for key, value in summary.items():
            f.write(f"- **{key}:** {value}\n")

        f.write("\n## Recent Runs\n")
        if not df.empty:
            f.write(df.tail(5).to_markdown(index=False))
        else:
            f.write("_No data available_\n")

        f.write("\n\n## Visual Analysis\n")
        if plots:
            for img in plots:
                f.write(img + "\n\n")
        else:
            f.write("_No plots generated_\n")

    log(f"Markdown summary written to: {summary_md}")

except Exception as e:
    log(f"[ERROR] Failed to write summary: {e}")

# === STEP 5: PRINT CONFIRMATION ===
log(f"Report saved at: {summary_md}")
log("=== Finished Summary Generation ===")