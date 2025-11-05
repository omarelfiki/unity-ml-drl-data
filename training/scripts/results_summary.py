"""
Generates a Markdown summary report for training runs.

Reads data/combined_results.csv, calculates average stats, creates plots,
and writes the report to data/summary_report.md.

DO NOT CHANGE FILE NAME OR LOCATION. Dependency for Github Actions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
from datetime import datetime

# SIMPLE LOGGING function
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

log("=== Starting Summary Generation ===")

# Define file paths relative to this script's location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
data_dir = os.path.join(project_root, "data")
csv_path = os.path.join(data_dir, "combined_results.csv")
summary_md = os.path.join(data_dir, "summary_report.md")

# Define key column names
TIMESTAMP_COL = None
ENV_COL = 'Environment'
NUMERIC_COLS_FOR_STATS = [
    "Steps", "Batch Size", "Buffer Size", "Learning Rate", "Epochs",
    "Total Time (s)", "Average CPU (%)", "Average RAM (%)",
    "Mean Policy Reward", "Mean Policy Loss", "Mean Value Loss", "Mean Entropy",
    "Threshold Steps", "Threshold Time (s)"
]
PLOT_PAIRS = [ # Column pairs for generating scatter plots
    ("Total Time (s)", "Mean Policy Reward"),
    ("Average CPU (%)", "Average RAM (%)"),
    ("Steps", "Mean Policy Reward")
]

# Log paths being used
log(f"Project root: {project_root}")
log(f"Data directory: {data_dir}")
log(f"Input CSV: {csv_path}")
log(f"Output Markdown: {summary_md}")

# Reads the CSV file into a pandas DataFrame and converts specified columns to numeric types.
df = None
if not os.path.exists(csv_path): #If csv file doesn't exist
     log(f"[ERROR] CRITICAL: CSV file not found at: {csv_path}. Cannot generate report.")
     df = pd.DataFrame() # Use an empty DataFrame if file is missing
else:
    try:
        df = pd.read_csv(csv_path)
        log(f"Successfully loaded data from CSV")

        # Converts columns to actual numeric types (float/int)
        converted_count = 0
        for col in NUMERIC_COLS_FOR_STATS:
            if col in df.columns: #Makes sure the array of column names matches the CSV columns
                if pd.api.types.is_numeric_dtype(df[col]): # Skip if already numeric
                    continue
                df[col] = pd.to_numeric(df[col], errors="coerce") #Handles NaN values
                converted_count += 1
            else:
                log(f"[WARNING] Expected numeric column '{col}' not found in CSV.")
        if converted_count > 0:
            log(f"Converted {converted_count} columns to numeric (errors -> NaN).")

        # Sort by timestamp to display 'recent runs' in report
        if TIMESTAMP_COL and TIMESTAMP_COL in df.columns: #Is timestamp not None, and does a col in df exist
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')
            df.dropna(subset=[TIMESTAMP_COL], inplace=True) # Remove rows with invalid dates
            df.sort_values(by=TIMESTAMP_COL, ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            log(f"Sorted data by timestamp column '{TIMESTAMP_COL}'.")
        else:
             log(f"[WARNING] No valid timestamp column ('{TIMESTAMP_COL}') found/configured. Using CSV order for 'recent runs'.")

        log("Data loaded and prepared.")

    except Exception as e:
        log(f"[ERROR] Failed during CSV loading or preparation: {e}")
        df = pd.DataFrame() # Ensure df is empty on error

# Calculates stats: total runs, unique environments, and averages for numeric columns.
summary_stats = {}
if df is None or df.empty:
     log("[WARNING] DataFrame empty. Limited statistics available.")
     summary_stats['total_runs'] = 0
     summary_stats['unique_environments'] = 0
     summary_stats['averages'] = {}
else:
    # Count total runs and unique environments
    summary_stats['total_runs'] = len(df)
    summary_stats['unique_environments'] = df[ENV_COL].nunique() if ENV_COL in df.columns else 'N/A'

    # Calculate averages for numeric columns, skipping NaN values
    summary_stats['averages'] = {}
    log("Calculating average statistics...")
    calculated_avg_count = 0
    for col in NUMERIC_COLS_FOR_STATS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            avg = df[col].mean(skipna=True) # Calculates mean, ignoring NaNs
            if pd.notna(avg): # Check if calculation resulted in a valid number
                 summary_stats['averages'][col] = avg
                 calculated_avg_count += 1
            else:
                 summary_stats['averages'][col] = 'N/A (No valid data)'
                 log(f"[INFO] No valid data to calculate average for '{col}'.")
        elif col in df.columns: # Exists but not numeric
             summary_stats['averages'][col] = 'N/A (Not numeric)'

    log(f"Computed averages for {calculated_avg_count} numeric columns.")


# Defines function to create scatter plots and converts them to Base64 strings.
def create_chart_base64(df_plot, x_col, y_col, title):
    # Check if input data and columns are valid for plotting
    if df_plot is None or df_plot.empty: return None
    if not all(c in df_plot.columns for c in [x_col, y_col]): return None
    if not all(pd.api.types.is_numeric_dtype(df_plot[c]) for c in [x_col, y_col]): return None

    # Prepare data by removing rows with NaN in either column
    plot_data = df_plot[[x_col, y_col]].dropna()
    if plot_data.empty:
        log(f"[WARNING] No valid data points for plot '{title}'.")
        return None

    try:
        plt.figure(figsize=(8, 4))
        plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, s=15)
        plt.title(title)
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel(y_col.replace("_", " ").title())
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Ensure plots directory exists
        plots_dir = (project_root + "\data\plots")
        os.makedirs(plots_dir, exist_ok=True)
        file_path = os.path.join(plots_dir, f"{title}.png")

        # Save plot directly to disk
        plt.savefig(file_path, dpi=90)
        plt.close()  # Free memory

        # Encode to Base64 string from saved file
        with open(file_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        log(f"Generated plot: {title} (saved to {file_path})")
        return f"![{title}](data:image/png;base64,{img_base64})"

    except Exception as e:
        log(f"[ERROR] Error creating plot '{title}': {e}")
        plt.close()  # Ensure figure is closed on error
        return None

# Generate the plots defined in PLOT_PAIRS
plots_md = []
if df is not None and not df.empty:
    log(f"Generating {len(PLOT_PAIRS)} plot(s)...")
    for x_col, y_col in PLOT_PAIRS:
        plot_title = f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}'
        chart_md = create_chart_base64(df, x_col, y_col, plot_title)
        if chart_md: plots_md.append(chart_md) # Collect successful plot strings
else:
     log("[WARNING] Skipping plot generation as DataFrame is empty.")
log(f"Successfully generated {len(plots_md)} plots.")

# Helper function to format numbers clearly in the report
def format_number(num):
    if isinstance(num, (int, float)):
         if pd.isna(num): return 'N/A'
         return f"{num:,.2f}" if isinstance(num, float) else f"{num:,}"
    return str(num)

# Writes the report contents to a Markdown file.
log("Generating Markdown report content...")
report_parts = [] # Build the report as a list of strings
try:
    # Header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_parts.append("# Training Summary Report")
    report_parts.append(f"`Report generated on: {now}`\n")
    report_parts.append("Summary of training run results.\n")
    report_parts.append("---")

    # Overview
    report_parts.append("## Overview Statistics")
    if not summary_stats or summary_stats.get('total_runs', 0) == 0 :
        report_parts.append("_No statistics available._")
    else:
        report_parts.append(f"- **Total Runs:** {summary_stats.get('total_runs', 'N/A')}")
        report_parts.append(f"- **Unique Environments:** {summary_stats.get('unique_environments', 'N/A')}\n")

        # Averages Table
        report_parts.append("### Key Metrics (Averages)")
        avg_stats = summary_stats.get('averages', {})
        if avg_stats:
            report_parts.append("| Metric                     | Average Value     |")
            report_parts.append("| :------------------------- | :---------------- |")
            for col, avg in avg_stats.items():
                report_parts.append(f"| {col.replace('_', ' ').title():<26} | {format_number(avg):<17} |") # Format table row
            report_parts.append("") #Add space after table
        else:
            report_parts.append("_No average metrics calculated._")

    report_parts.append("\n---")

    # Recent Runs Section
    report_parts.append("## Recent Runs (Last 5)")
    if df is None or df.empty:
        report_parts.append("_No run data available._")
    else:
        # Define columns to show in the recent runs table
        recent_cols_priority = [ENV_COL, "Mean Policy Reward", "Total Time (s)", "Steps", "Average CPU (%)"]
        display_cols = [col for col in recent_cols_priority if col in df.columns]

        # Select the last 5 rows based on CSV order (since no timestamp sort is guaranteed)
        if TIMESTAMP_COL is None:
             recent_runs_df = df.tail(5)[display_cols].copy().iloc[::-1] # Get last 5, reverse order
        else:
             recent_runs_df = df.head(5)[display_cols].copy() # Get first 5 (already sorted if timestamp existed)

        if not recent_runs_df.empty:
             # Format numeric columns for display
             for col in recent_runs_df.select_dtypes(include=['number']).columns:
                 if col in recent_runs_df:
                     recent_runs_df[col] = recent_runs_df[col].apply(format_number)
             # Format timestamp if present
             if TIMESTAMP_COL and TIMESTAMP_COL in recent_runs_df.columns and pd.api.types.is_datetime64_any_dtype(recent_runs_df[TIMESTAMP_COL]):
                  recent_runs_df[TIMESTAMP_COL] = recent_runs_df[TIMESTAMP_COL].dt.strftime('%Y-%m-%d %H:%M')

             # Replace any remaining NaN values with 'N/A' string for the table
             recent_runs_df.fillna('N/A', inplace=True)
             # Convert the formatted DataFrame to Markdown table format
             report_parts.append(recent_runs_df.to_markdown(index=False))
        else:
             report_parts.append("_No recent runs to display._")

    report_parts.append("\n---")

    # Plots Section
    report_parts.append("## Trend Plots")
    if plots_md:
        for img_md in plots_md:
            title_start = img_md.find('[') + 1
            title_end = img_md.find(']')
            plot_title = img_md[title_start:title_end] if title_start > 0 and title_end > title_start else "Plot"
            report_parts.append(f"### {plot_title}\n")
            report_parts.append(img_md)
            report_parts.append("")
    else:
        report_parts.append("_No plots generated._")

    #TODO Implement automated insights and anomaly detection
    report_parts.append("\n---")
    report_parts.append("## Key Insights & Anomalies")
    report_parts.append("_(Automated insights generation not implemented)_")

    # Combine report parts and write to file
    final_report_content = "\n".join(report_parts)
    log("Markdown content generated.")

    log(f"Writing report to {summary_md}...")
    try:
        os.makedirs(os.path.dirname(summary_md), exist_ok=True) # Ensure directory exists
        with open(summary_md, "w", encoding='utf-8') as f:
            f.write(final_report_content)
        log(f"Successfully wrote summary report.")
    except IOError as e:
        log(f"[ERROR] Failed to write summary report file: {e}")

except Exception as e:
     log(f"[ERROR] An unexpected error occurred during report generation: {e}")

log(f"Report generation process finished. Final report should be at: {summary_md}")
log("=== Finished Summary Generation ===")