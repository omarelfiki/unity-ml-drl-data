"""Display and output utilities."""
import csv
import json
import os
import shutil
import time
import re

from scripts.models.data_models import TrainingArgs
from scripts.config.constants import VERSION, CSV_HEADERS, KEY_MAPPING


def print_intro(args: TrainingArgs):
    title = f"ML-Agents Training Runner V{VERSION} by AIML 6 - Maastricht University DACS Project 2-1: Artificial Intelligence and Machine Learning"

    if args.seed is not None:
        seed = args.seed
    elif args.batch_range is not None:
        start, end = args.batch_range
        seed = f"{start}-{end}"
    else:
        seed = "Undefined"

    items = [
        f"Run ID: {args.run_id}",
        f"Config: {os.path.basename(args.config) if args.config else 'N/A'}",
        f"Auto-commit: {'ON' if args.ac else 'OFF'}",
        f"Env: {args.env_path if args.env_path else 'Unity Editor'}",
        f"Seed: {seed}",
        f"Steps: {args.num_steps if args.num_steps is not None else 'Defined in Config'}",
        f"Thresholds: {'ON' if not args.no_thresholds else 'OFF'}",
        f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Verbose: {args.verbose if args.verbose is not None else 'Disabled'}",
    ]

    joined = "  \u2022  ".join(items)
    visible_len = len(strip_ansi(joined))
    term_width = shutil.get_terminal_size((120, 20)).columns
    min_inner = 60
    inner_width = max(min_inner, visible_len)
    inner_width = min(inner_width, max(10, term_width - 4))

    # Center data line
    pad_total = max(0, inner_width - visible_len)
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad
    padded_meta = " " * left_pad + joined + " " * right_pad
    title_len = len(strip_ansi(title))
    title_pad_total = max(0, inner_width - title_len)
    title_left = title_pad_total // 2
    title_right = title_pad_total - title_left
    padded_title = " " * title_left + title + " " * title_right

    print(f"| {padded_title} |")
    print(f"| {padded_meta} |")

def strip_ansi(s: str) -> str:
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_re.sub("", s)

def save_and_display_results(combined_data: dict, v: bool = False):
    key_width = max(len(k) for k in combined_data.keys())
    val_width = max(len(v) for v in combined_data.values())

    print("\n" + "=" * (key_width + val_width + 7))
    print(f"| {'Metric'.ljust(key_width)} | {'Value'.ljust(val_width)} |")
    print("=" * (key_width + val_width + 7))

    for key, value in combined_data.items():
        print(f"| {key.ljust(key_width)} | {value.ljust(val_width)} |")

    print("=" * (key_width + val_width + 7))

    # Save results to CSV and JSON in 'data' folder outside 'training' directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_file = os.path.join(data_dir, "combined_results.csv")
    json_file = os.path.join(data_dir, "combined_results.json")

    normalized_data = {}
    for old_key, new_key in KEY_MAPPING.items():
        if old_key in combined_data:
            normalized_data[new_key] = combined_data[old_key]
        else:
            normalized_data[new_key] = ""

    # Write or append to CSV
    file_exists = os.path.isfile(csv_file)
    # Ensure the file ends with a newline before appending
    with open(csv_file, 'a+', newline='') as csvfile:
        csvfile.seek(0, os.SEEK_END)
        if csvfile.tell() > 0:
            csvfile.seek(csvfile.tell() - 1)
            last_char = csvfile.read(1)
            if last_char != '\n':
                csvfile.write('\n')

        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        filtered_data = {key: normalized_data.get(key, "") for key in CSV_HEADERS}
        writer.writerow(filtered_data)
        if v: print(f"[INFO] Results saved to '{csv_file}'")

    # Write or append to JSON
    json_data = []
    if os.path.isfile(json_file):
        try:
            with open(json_file, 'r') as jf:
                json_data = json.load(jf)
        except Exception:
            json_data = []

    json_data.append(combined_data)
    with open(json_file, 'w') as jf:
        json.dump(json_data, jf, indent=4)
        if v: print(f"[INFO] Results saved to '{json_file}'")