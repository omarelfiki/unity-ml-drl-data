import json
import numpy as np
from joblib import dump
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common_features import load_df, make_preprocess, DATA


def train_pipeline(df, test_size=0.2, seed=42, thresh=0.5):
    pre, feats = make_preprocess(df)
    X, y = df[feats], df["run_reached_threshold"].astype(int).values

    # One shared split for both stages
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(
        X, y, df, test_size=test_size, random_state=seed, stratify=y
    )

    # Stage 1: reach classifier
    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= thresh).astype(int)

    clf_metrics = {
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "features": feats,
        "threshold": float(thresh),
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }

    # Stage 2: regressors trained only on reached runs (train split)
    dtr_r = dtr[(dtr["run_reached_threshold"] == 1) &
               (dtr["steps_to_threshold"] > 0) & (dtr["time_to_threshold"] > 0)].copy()
    if len(dtr_r) < 5:
        raise ValueError("Not enough reached runs in train split to fit regressors.")

    base = Pipeline([("pre", pre), ("lr", LinearRegression())])
    ytfm = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
    mk = lambda: TransformedTargetRegressor(regressor=base, transformer=ytfm, check_inverse=False)

    m_steps, m_time = mk(), mk()
    m_steps.fit(dtr_r[feats], dtr_r["steps_to_threshold"].values)
    m_time.fit(dtr_r[feats], dtr_r["time_to_threshold"].values)

    # Evaluate regressors on reached-only test rows (conditional evaluation)
    dte_r = dte[(dte["run_reached_threshold"] == 1) &
               (dte["steps_to_threshold"] > 0) & (dte["time_to_threshold"] > 0)].copy()

    reg_metrics = {"n_test_reached": int(len(dte_r))}
    if len(dte_r) >= 1:
        ps = np.clip(m_steps.predict(dte_r[feats]), 0, np.inf)
        pt = np.clip(m_time.predict(dte_r[feats]), 0, np.inf)
        reg_metrics["steps"] = {
            "mae": float(mean_absolute_error(dte_r["steps_to_threshold"], ps)),
            "rmse": float(np.sqrt(mean_squared_error(dte_r["steps_to_threshold"], ps))),
            "r2": float(r2_score(dte_r["steps_to_threshold"], ps)) if len(dte_r) >= 2 else float("nan"),
        }
        reg_metrics["time"] = {
            "mae": float(mean_absolute_error(dte_r["time_to_threshold"], pt)),
            "rmse": float(np.sqrt(mean_squared_error(dte_r["time_to_threshold"], pt))),
            "r2": float(r2_score(dte_r["time_to_threshold"], pt)) if len(dte_r) >= 2 else float("nan"),
        }

    # Build a predictions file on the held-out test split
    out = dte.copy()
    out["p_reach"] = proba
    out["pred_reach"] = pred
    out["pred_steps_to_threshold_cond"] = np.clip(m_steps.predict(out[feats]), 0, np.inf)
    out["pred_time_to_threshold_cond"] = np.clip(m_time.predict(out[feats]), 0, np.inf)

    # Expected steps (failure cost = max steps budget)
    if "steps" in out.columns:
        out["expected_steps"] = out["p_reach"] * out["pred_steps_to_threshold_cond"] + (1 - out["p_reach"]) * out["steps"]

    metrics = {"classifier": clf_metrics, "regressors": reg_metrics}
    models = {"logistic": clf, "linear_steps": m_steps, "linear_time": m_time}
    return models, metrics, out


def main():
    df = load_df(DATA)
    models, metrics, preds = train_pipeline(df)

    dump(models["logistic"], "logistic_reach_model.joblib")
    dump(models["linear_steps"], "linear_steps_model.joblib")
    dump(models["linear_time"], "linear_time_model.joblib")

    with open("predict_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    preds.to_csv("test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nSaved: logistic_reach_model.joblib, linear_steps_model.joblib, linear_time_model.joblib, "
          "predict_metadata.json, test_predictions.csv")


if __name__ == "__main__":
    main()
