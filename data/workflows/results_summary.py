"""
Generates a Markdown summary report for training runs.

Reads data/collected_results.csv, calculates stats (mean, std),
creates plots, finds insights/anomalies, and writes the report
to data/summary_report.md.

Dependency for GitHub Actions workflows on repo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
from datetime import datetime, timezone
from paths import REPORTS_DIR, CSV_FILE, PLOTS_DIR

CSV_HEADERS = [
    "run_id", "environment", "seed", "num_agents",
    "algorithm", "steps", "batch_size", "buffer_size",
    "learning_rate", "epochs",
    "total_time", "average_cpu", "average_ram",
    "reward_mean", "reward_mean_step", "early_reward_mean", "early_reward_mean_step",
    "final_reward_mean", "final_reward_mean_step", "best_reward", "best_reward_step", "step_interval",
    "p_loss_mean", "p_loss_mean_step", "v_loss_mean", "v_loss_mean_step",
    "entropy_mean", "entropy_mean_step",
    "threshold_value", "steps_to_threshold", "time_to_threshold", "threshold_version", "run_reached_threshold",
]

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

log("=== Starting Summary Generation ====")
SUMMARY_MD = os.path.join(REPORTS_DIR, "summary_report.md")

TIMESTAMP_COL = None
ENV_COL = "environment"
ID_COL = "run_id"

NUMERIC_COLS_FOR_STATS = [
    "num_agents", "steps", "batch_size", "buffer_size", "learning_rate", "epochs",
    "total_time", "average_cpu", "average_ram",
    "step_interval", "reward_mean", "reward_mean_step", "early_reward_mean", "early_reward_mean_step",
    "final_reward_mean", "final_reward_mean_step", "best_reward", "best_reward_step",
    "p_loss_mean", "p_loss_mean_step", "v_loss_mean", "v_loss_mean_step",
    "entropy_mean", "entropy_mean_step",
    "threshold_value", "steps_to_threshold", "time_to_threshold", "threshold_version",
    "run_reached_threshold",
]

PLOT_PAIRS = [
    ("total_time", "reward_mean"),
    ("average_cpu", "average_ram"),
    ("steps", "reward_mean"),
]

KEY_INSIGHT_COLS = ["reward_mean", "p_loss_mean", "average_cpu", "total_time"]

log(f"Input CSV: {CSV_FILE}")
log(f"Output Markdown: {SUMMARY_MD}")

# Load data
df = pd.DataFrame()
if os.path.exists(CSV_FILE):
    try:
        df = pd.read_csv(CSV_FILE)
        log("Successfully loaded data from CSV")

        # Convert numeric columns
        for col in NUMERIC_COLS_FOR_STATS:
            if col not in df.columns:
                log(f"[WARNING] Configured numeric column '{col}' not found in CSV.")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by timestamp if configured
        if TIMESTAMP_COL is not None and TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
            df = df.dropna(subset=[TIMESTAMP_COL])
            df = df.sort_values(by=TIMESTAMP_COL, ascending=False).reset_index(drop=True)
            log(f"Sorted data by '{TIMESTAMP_COL}'.")
        else:
            log("[INFO] No timestamp column configured; using CSV order for 'recent' runs.")

        log("Data loaded and prepared.")
    except Exception as e:
        log(f"[ERROR] Failed during CSV loading: {e}")
else:
    log(f"[ERROR] CSV file not found at: {CSV_FILE}.")

# Compute statistics
summary_stats = {}
if not df.empty:
    summary_stats["total_runs"] = len(df)
    summary_stats["unique_environments"] = df[ENV_COL].nunique() if ENV_COL in df.columns else 0

    averages = {}
    stds = {}
    for col in NUMERIC_COLS_FOR_STATS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            avg = df[col].mean(skipna=True)
            std = df[col].std(skipna=True)
            if pd.notna(avg):
                averages[col] = avg
                stds[col] = std
    summary_stats["averages"] = averages
    summary_stats["stds"] = stds
    log(f"Computed stats for {len(averages)} numeric columns.")
else:
    summary_stats = {
        "total_runs": 0,
        "unique_environments": 0,
        "averages": {},
        "stds": {},
    }

def create_chart_base64(df_plot, x_col, y_col, title):
    if df_plot is None or df_plot.empty:
        return None
    if not all(c in df_plot.columns for c in [x_col, y_col]):
        return None
    if not all(pd.api.types.is_numeric_dtype(df_plot[c]) for c in [x_col, y_col]):
        return None

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

        os.makedirs(PLOTS_DIR, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(PLOTS_DIR, f"{safe_title}.png")

        plt.savefig(file_path, dpi=90)
        plt.close()

        with open(file_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        log(f"Generated plot: {title} (saved to {file_path})")
        return f"![{title}](data:image/png;base64,{img_base64})"

    except Exception as e:
        log(f"[ERROR] Error creating plot '{title}': {e}")
        plt.close()
        return None

plots_md = []
if not df.empty:
    for x_col, y_col in PLOT_PAIRS:
        plot_title = f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}'
        chart_md = create_chart_base64(df, x_col, y_col, plot_title)
        if chart_md:
            plots_md.append(chart_md)

def format_number(num):
    if pd.isna(num):
        return "N/A"
    if isinstance(num, (int, float)):
        return f"{int(num):,}" if num % 1 == 0 else f"{num:,.2f}"
    return str(num)

def generate_insights(df, summary_stats):
    insights = []
    if df.empty:
        return ["No data available for insights."]

    averages = summary_stats.get("averages", {})
    stds = summary_stats.get("stds", {})
    reward_col = "reward_mean"

    if reward_col in df.columns and pd.notna(df[reward_col].max()):
        best_run_idx = df[reward_col].idxmax()
        best_run = df.loc[best_run_idx]
        insights.append(
            f"**Top Performer:** Best run (highest reward: {format_number(best_run[reward_col])}) "
            f"was for Environment '{best_run.get(ENV_COL, 'N/A')}' "
            f"(Run ID: `{best_run.get(ID_COL, 'N/A')}`)."
        )

    anomalies_found = False
    for col in KEY_INSIGHT_COLS:
        if col in averages and col in stds and pd.notna(stds[col]) and stds[col] > 0:
            mean, std = averages[col], stds[col]
            upper, lower = mean + 3 * std, mean - 3 * std
            outliers = df[(df[col] > upper) | (df[col] < lower)]
            if not outliers.empty:
                anomalies_found = True
                idx = (outliers[col] - mean).abs().idxmax()
                max_dev = outliers.loc[idx, col]
                insights.append(
                    f"**Anomaly ({col.replace('_', ' ').title()}):** Found **{len(outliers)} run(s)** outside 3σ "
                    f"(Mean: {format_number(mean)}, Std: {format_number(std)}). "
                    f"Most extreme value: {format_number(max_dev)}."
                )

    if not anomalies_found:
        insights.append("No significant anomalies detected (all values within 3 standard deviations).")

    if reward_col in averages:
        recent_avg = df.tail(5)[reward_col].mean()
        overall_avg = averages[reward_col]
        if pd.notna(recent_avg) and pd.notna(overall_avg):
            if recent_avg > overall_avg * 1.1:
                insights.append(
                    f"**Recent Trend:** Performance is **improving**. Last 5 runs avg = "
                    f"{format_number(recent_avg)} (>10% higher than overall {format_number(overall_avg)})."
                )
            elif recent_avg < overall_avg * 0.9:
                insights.append(
                    f"**Recent Trend:** Performance is **declining**. Last 5 runs avg = "
                    f"{format_number(recent_avg)} (>10% lower than overall {format_number(overall_avg)})."
                )
            else:
                insights.append(
                    f"**Recent Trend:** Performance is **stable** (within ±10% of average)."
                )

    return insights or ["No key insights or anomalies found."]

log("Generating Markdown report content...")
report_parts = []
try:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_parts.append("# Training Summary Report")
    report_parts.append(f"`Report generated on: {now}`\n")
    report_parts.append("Summary of training run results.\n")

    report_parts.append("Overview Statistics")
    if summary_stats["total_runs"] == 0:
        report_parts.append("_No statistics available._")
    else:
        report_parts.append(f"- **Total Runs:** {summary_stats['total_runs']}")
        report_parts.append(f"- **Unique Environments:** {summary_stats['unique_environments']}\n")

        avg_stats = summary_stats["averages"]
        if avg_stats:
            report_parts.append("### Key Metrics (Averages)")
            report_parts.append("| Metric | Average Value |")
            report_parts.append("| :------ | :------------- |")
            for col, avg in sorted(avg_stats.items()):
                report_parts.append(
                    f"| {col.replace('_', ' ').title()} | {format_number(avg)} |"
                )

        std_stats = summary_stats["stds"]
        if std_stats:
            report_parts.append("\n### Key Metrics (Standard Deviations)")
            report_parts.append("| Metric | Std. Deviation |")
            report_parts.append("| :------ | :------------- |")
            for col, std in sorted(std_stats.items()):
                if pd.notna(std):
                    report_parts.append(
                        f"| {col.replace('_', ' ').title()} | {format_number(std)} |"
                    )

    report_parts.append("\nRecent Runs (Last 5)")
    if df.empty:
        report_parts.append("_No run data available._")
    else:
        recent_cols = [
            c
            for c in [ENV_COL, "reward_mean", "total_time", "steps", "average_cpu"]
            if c in df.columns
        ]
        recent_df = df.tail(5)[recent_cols].iloc[::-1].copy()
        for col in recent_df.select_dtypes(include=["number"]).columns:
            recent_df[col] = recent_df[col].apply(format_number)
        recent_df = recent_df.fillna("N/A")
        report_parts.append(recent_df.to_markdown(index=False))

    report_parts.append("\nTrend Plots")
    if plots_md:
        for img_md in plots_md:
            title = img_md.split("[")[1].split("]")[0]
            report_parts.append(f"### {title}\n{img_md}\n")
    else:
        report_parts.append("_No plots generated._")

    report_parts.append("\nKey Insights & Anomalies")
    log("Analyzing data for insights...")
    for insight in generate_insights(df, summary_stats):
        report_parts.append(f"- {insight}")

    os.makedirs(os.path.dirname(SUMMARY_MD), exist_ok=True)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(report_parts))
    log("Successfully wrote summary report.")
except Exception as e:
    log(f"[ERROR] Error during report generation: {e}")

log("=== Finished Summary Generation ===")
