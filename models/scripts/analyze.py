import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, mean_absolute_error, roc_auc_score
import pandas as pd

def analyze(exp_path, actual, env= None, skipped=None):
    df_all = pd.read_csv(actual)
    version = actual.split("_")[-1].split(".")[0]
    analysis_dir = "experiments" / Path(exp_path) / "analysis"
    total = len(df_all)
    env_name = env if env else "all"
    js = {"name": f"{env_name} on dataset", "version": version, "total_samples": total,
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
        plots_dir = adir / "plots"
        plots_dir.mkdir(parents=True)
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
        plot_confusion_matrix(cmx, env, plots_dir / "confusion_matrix.png")
        plot_roc(df["run_reached_threshold"], df["pred_reach"], env, plots_dir / "roc.png")
        plot_regression_errors(df, env, plots_dir)
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

    # undefined if only one class
    if actual.nunique() < 2:
        return float("nan")

    return roc_auc_score(actual, predicted)

def plot_confusion_matrix(cmx, env, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow([[cmx["TN"], cmx["FP"]],
                    [cmx["FN"], cmx["TP"]]])

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title(f"Confusion Matrix – {env}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    [[cmx["TN"], cmx["FP"]],
                     [cmx["FN"], cmx["TP"]]][i][j],
                    ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_roc(actual, predicted, env, out_path):
    if actual.nunique() < 2:
        return  # ROC undefined

    fpr, tpr, _ = roc_curve(actual, predicted)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {env}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_regression_errors(df, env, out_dir):
    mask = df["run_reached_threshold"] == 1
    if mask.sum() == 0:
        return

    # steps
    mae_steps = mean_absolute_error(
        df.loc[mask, "steps_to_threshold"],
        df.loc[mask, "pred_steps_to_threshold_cond"]
    )

    # time
    mae_time = mean_absolute_error(
        df.loc[mask, "time_to_threshold"],
        df.loc[mask, "pred_time_to_threshold_cond"]
    )

    # bar
    fig, ax = plt.subplots()
    ax.bar(["Steps MAE", "Time MAE"], [mae_steps, mae_time])
    ax.set_title(f"Regression MAE – {env}")
    ax.set_ylabel("Error")

    fig.tight_layout()
    fig.savefig(out_dir / "regression_mae.png")
    plt.close(fig)

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
