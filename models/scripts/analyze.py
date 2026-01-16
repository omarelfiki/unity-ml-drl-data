import json
from pathlib import Path
import pandas as pd

def analyze(exp_path, actual, env= None, skipped=None):
    df_all = pd.read_csv(actual)
    analysis_dir = "experiments" / Path(exp_path) / "analysis"
    total = len(df_all)
    js = {"environment": env if env else "all", "total_samples": total,
          "samples per environment": df_all["environment"].value_counts().to_dict(),
          "% reaching threshold": len([df_all["run_reached_threshold"] == 1]) / len(df_all) * 100}
    metadata_file = Path(analysis_dir) / "metadata.json"
    json.dump(js, metadata_file.open("w"), indent=2)

    envs = sorted(df_all["environment"].dropna().unique().tolist())
    if skipped:
        for skip in skipped:
            envs.remove(skip)
    if env is not None:
        envs = [env]
    for env in envs:
        predicted_file = "experiments" / Path(exp_path) / "predictions" / f"{env}" / "models_prediction.csv"
        predicted = Path(predicted_file)
        df = pd.read_csv(predicted)
        print(f"[INFO] Predictions: {predicted_file}")
        verify_data(df)
        print(f"[INFO] Predictions Validated")
        adir = analysis_dir / env
        adir.mkdir(parents=True)
        analysis_file = adir / "analysis.csv"
        df.to_csv(analysis_file, index=False)
    return

def verify_data(df):
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

    for col in [c for c in required if c not in {"time_to_threshold", "steps_to_threshold"}]:
        if df[col].isna().any():
            raise AssertionError(f"Column `{col}` contains null values")
