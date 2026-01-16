import csv
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score

FEATURE_COLUMNS = [
    "seed",
    "num_agents",
    "batch_size",
    "buffer_size",
    "learning_rate",
    "epochs",
    "average_cpu",
    "average_ram",
    "step_interval",
    "p_loss_mean",
    "p_loss_mean_step",
    "v_loss_mean",
    "v_loss_mean_step",
    "entropy_mean",
    "entropy_mean_step",
    "early_reward_mean",
    "early_reward_mean_step",
    "final_reward_mean",
    "final_reward_mean_step",
]


def build_classifier(seed):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1
    )


def build_regressors(seed):
    reg_steps = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=seed,
        n_jobs=-1
    )

    reg_time = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

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

        clf = build_classifier(seed)

        X_train_clf = train_df[FEATURE_COLUMNS]
        y_train_clf = train_df["target_class"]

        clf.fit(X_train_clf, y_train_clf)

        # Predict on the test set
        X_test_clf = test_df[FEATURE_COLUMNS]
        y_test_clf = test_df["target_class"]
        y_pred_clf = clf.predict(X_test_clf)

        clf_metrics = {
            "accuracy": accuracy_score(y_test_clf, y_pred_clf),
            "f1": f1_score(y_test_clf, y_pred_clf, zero_division=0),
        }

        train_r = train_df[train_df["run_reached_threshold"] == 1]
        test_r = test_df[test_df["run_reached_threshold"] == 1]

        # Define minimum reached threshold constant
        MIN_REACHED = 10

        if len(train_r) < MIN_REACHED:
            continue

        reg_steps, reg_time = build_regressors(seed)

        X_train_reg = train_r[FEATURE_COLUMNS]
        y_train_steps = train_r["steps"]
        y_train_time = train_r["total_time"]

        reg_steps.fit(X_train_reg, y_train_steps)
        reg_time.fit(X_train_reg, y_train_time)

        X_test_reg = test_r[FEATURE_COLUMNS]
        y_test_steps = test_r["steps"]
        y_test_time = test_r["total_time"]

        y_pred_steps = reg_steps.predict(X_test_reg)
        y_pred_time = reg_time.predict(X_test_reg)

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