import json
import numpy as np
import argparse
from joblib import dump
from sklearn.model_selection import train_test_split

from common_features import load_df, DATA
import logistic as log
import linear as lin


def main(test_size=0.2, seed=42, thresh=0.5):
    df = load_df(DATA)

    # One shared split for both stages
    y = df["run_reached_threshold"].astype(int).values
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)

    # Stage 1: logistic (reach probability)
    clf, clf_metrics, proba, pred, feats = log.train_classifier_split(train_df, test_df, thresh=thresh)

    # Stage 2: linear (conditional time/steps)
    m_steps, m_time, reg_metrics, feats2 = lin.train_two_regressors_split(train_df, test_df)

    # Feature lists should match; if not, merge safely
    feats_use = feats if feats == feats2 else list(dict.fromkeys(feats + feats2))

    out = test_df.copy()
    out["p_reach"] = proba
    out["pred_reach"] = pred
    out["pred_steps_to_threshold_cond"] = np.clip(m_steps.predict(out[feats_use]), 0, np.inf)
    out["pred_time_to_threshold_cond"] = np.clip(m_time.predict(out[feats_use]), 0, np.inf)

    if "steps" in out.columns:
        out["expected_steps"] = out["p_reach"] * out["pred_steps_to_threshold_cond"] + (1 - out["p_reach"]) * out["steps"]

    metrics = {"classifier": clf_metrics, "regressors": reg_metrics}
    with open("predict_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    out.to_csv("test_predictions.csv", index=False)

    # Save models for later inference
    dump(clf, "logistic_reach_model.joblib")
    dump(m_steps, "linear_steps_model.joblib")
    dump(m_time, "linear_time_model.joblib")

    print(json.dumps(metrics, indent=2))
    print("\nSaved: logistic_reach_model.joblib, linear_steps_model.joblib, linear_time_model.joblib, "
          "predict_metadata.json, test_predictions.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage prediction pipeline for 3DBall.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data used as test set (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splitting (default: 42).")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Threshold for pred_reach from p_reach (default: 0.5).")

    args = parser.parse_args()
    main(test_size=args.test_size, seed=args.seed, thresh=args.thresh)