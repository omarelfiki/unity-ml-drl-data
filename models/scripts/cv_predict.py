import csv
import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from scripts.common_features import NUM_FEATS

def build_classifier(seed):
    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
        class_weight="balanced",
        solver="lbfgs",
    )

def build_regressors(seed):
    reg_steps = LinearRegression(n_jobs=-1)
    reg_time = LinearRegression(n_jobs=-1)
    return reg_steps, reg_time


def make_groups(df):
    return (
            df["environment"].astype(str) + "_" +
            df["algorithm"].astype(str) + "_" +
            df["batch_size"].astype(str) + "_" +
            df["buffer_size"].astype(str) + "_" +
            df["learning_rate"].astype(str) + "_" +
            df["epochs"].astype(str)
    )


def run_cv(path, df, n_splits=5, seed=42):
    df = pd.read_csv(df)
    df["target_class"] = df["run_reached_threshold"].astype(int)
    groups = make_groups(df)
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []

    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups)):
        train_df = df.iloc[tr]
        test_df = df.iloc[te]

        try:
            clf = build_classifier(seed)
        except Exception as e:
            raise Exception(f"Failed to build classifier: {e}")

        X_train_clf = train_df[NUM_FEATS].fillna(0)
        y_train_clf = train_df["target_class"]

        clf.fit(X_train_clf, y_train_clf)

        # Predict on the test set
        X_test_clf = test_df[NUM_FEATS].fillna(0)
        y_test_clf = test_df["target_class"]
        y_pred_clf = clf.predict(X_test_clf)

        try:
            clf_metrics = {
            "accuracy": accuracy_score(y_test_clf, y_pred_clf),
            "f1": f1_score(y_test_clf, y_pred_clf, zero_division=0),
            }
        except Exception as e:
           raise Exception(f"Failed to compute metrics: {e}")

        train_r = train_df[
            (train_df["run_reached_threshold"] == 1) &
            train_df["steps_to_threshold"].notna() &
            train_df["time_to_threshold"].notna()
            ]

        test_r = test_df[
            (test_df["run_reached_threshold"] == 1) &
            test_df["steps_to_threshold"].notna() &
            test_df["time_to_threshold"].notna()
            ]

        # Define minimum reached threshold constant
        MIN_REACHED = 10

        if len(train_r) < MIN_REACHED:
            print("[WARNING]: Not enough reached runs in training split to fit regressors. Skipping fold.")
            continue

        reg_steps, reg_time = build_regressors(seed)

        X_train_reg = train_r[NUM_FEATS]
        y_train_steps = train_r["steps_to_threshold"]
        y_train_time = train_r["time_to_threshold"]

        try:
            reg_steps.fit(X_train_reg, y_train_steps)
            reg_time.fit(X_train_reg, y_train_time)
        except Exception as e:
            raise Exception(f"Failed to fit regressors: {e}")

        X_test_reg = test_r[NUM_FEATS]
        y_test_steps = test_r["steps_to_threshold"]
        y_test_time = test_r["time_to_threshold"]

        try:
            y_pred_steps = reg_steps.predict(X_test_reg)
            y_pred_time = reg_time.predict(X_test_reg)
        except Exception as e:
            raise Exception(f"Failed to predict with regressors: {e}")

        reg_metrics = {
            "steps_mse": mean_squared_error(y_test_steps, y_pred_steps),
            "steps_mae": mean_absolute_error(y_test_steps, y_pred_steps),
            "time_mse": mean_squared_error(y_test_time, y_pred_time),
            "time_mae": mean_absolute_error(y_test_time, y_pred_time),
        }

        fold_metrics.append({
            "fold": fold,
            "classifier": clf_metrics,
            "regression": reg_metrics
        })

    save_cv_results(fold_metrics, path)


def save_cv_results(results, path):
    cv_dir = "experiments" / Path(path) / "cv"

    with open(cv_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    out_path = cv_dir / f"folds.csv"

    fieldnames = [
        "fold",
        "accuracy",
        "f1",
        "steps_mse",
        "steps_mae",
        "time_mse",
        "time_mae",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "fold": r["fold"],
                "accuracy": r["classifier"]["accuracy"],
                "f1": r["classifier"]["f1"],
                "steps_mse": r["regression"]["steps_mse"],
                "steps_mae": r["regression"]["steps_mae"],
                "time_mse": r["regression"]["time_mse"],
                "time_mae": r["regression"]["time_mae"],
            }
            writer.writerow(row)