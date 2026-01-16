import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
import pandas as pd

def analyze(exp_path, actual, env= None, skipped=None):
    df_all = pd.read_csv(actual)
    analysis_dir = "experiments" / Path(exp_path) / "analysis"
    total = len(df_all)
    js = {"environment": env if env else "all", "total_samples": total,
          "samples per environment": df_all["environment"].value_counts().to_dict(),
          "% reaching threshold": (df_all["run_reached_threshold"] == 1).mean() * 100}
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
        cmx = confusion_matrix(df["run_reached_threshold"], df["pred_reach"])
        acc = accuracy(df["run_reached_threshold"], df["pred_reach"])
        prec = precision(df["run_reached_threshold"], df["pred_reach"])
        roc = roc_auc(df["run_reached_threshold"], df["pred_reach"])
        df_cmx = pd.DataFrame(
            [[cmx["TN"], cmx["FP"]],
             [cmx["FN"], cmx["TP"]]],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
        df_acc = pd.DataFrame([acc, prec, roc], index=["Accuracy", "Precision", "ROC AUC"], columns=[env])
        df_analysis = pd.concat([df_cmx, df_acc], axis=1)
        df_analysis.to_csv(analysis_file, index=True)

    return

def confusion_matrix(actual, predicted) -> dict:
    actual = actual.astype(int)
    predicted = predicted.astype(int)

    tp = ((actual == 1) & (predicted == 1)).sum()
    tn = ((actual == 0) & (predicted == 0)).sum()
    fp = ((actual == 0) & (predicted == 1)).sum()
    fn = ((actual == 1) & (predicted == 0)).sum()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

def accuracy(actual, predicted) -> float:
    actual = actual.astype(int)
    predicted = predicted.astype(int)
    return (actual == predicted).mean()

def precision(actual, predicted) -> float:
    actual = actual.astype(int)
    predicted = predicted.astype(int)

    tp = ((actual == 1) & (predicted == 1)).sum()
    fp = ((actual == 0) & (predicted == 1)).sum()

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def roc_auc(actual, predicted) -> float:
    actual = actual.astype(int)
    predicted = predicted.astype(int)

    # ROC AUC undefined if only one class
    if actual.nunique() < 2:
        return float("nan")

    return roc_auc_score(actual, predicted)

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
