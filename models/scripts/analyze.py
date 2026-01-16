import json
from pathlib import Path

import pandas as pd


def analyze(actual, predicted, env= None):
    predicted_file = Path(predicted)
    df = pd.read_csv(predicted_file)
    print(f"[INFO] Predictions: {predicted_file}")
    verify_data(df)
    print(f"[INFO] Predictions Validated")
    total = len(df)
    total_filtered = 0
    # if env:
    #     df = df[df["environment"] == env]
    #     total_filtered = total - len(df)
    # else:
    #     print("[INFO] No environment specified. Using all environments.")
    analysis_dir = predicted_file.parent.parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    analysis_file = analysis_dir / f"analysis.csv"
    metadata_file = analysis_dir / f"metadata.json"
    js = {"environment": env if env else "all", "total_samples": total, "filtered_samples": total_filtered, "samples per environment": df["environment"].value_counts().to_dict(), "% reaching threshold": len([df["pred_reach"] == 1]) / len(df) * 100}
    json.dump(js, metadata_file.open("w"), indent=2)
    df.to_csv(analysis_file, index=False)
    return

def verify_data(df):
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required = [
        "run_reached_threshold",
        "pred_reach",
        "steps_to_threshold",
        "pred_steps_to_threshold_cond",
        "time_to_threshold",
        "pred_time_to_threshold_cond",
        "environment",
    ]

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        cols_preview = list(df.columns)
        raise AssertionError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {cols_preview}"
        )

    for col in required:
        if df[col].isna().any():
            raise AssertionError(f"Column `{col}` contains null values")
