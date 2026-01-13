import json
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from datetime import datetime
from scripts.common_features import load_df
import scripts.logistic as log
import scripts.linear as lin
import scripts.utils as utils


def main(test_size=0.2, seed=42, thresh=0.5):
    print("[INFO]: Loading dataset splits and grouping features....")
    df = load_df()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # One shared split for both stages
    y = df["run_reached_threshold"].astype(int).values
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)

    print("[INFO]: Training models....")
    # Stage 1: logistic (reach probability)
    try:
        clf, clf_metrics, proba, pred, feats = log.train_classifier_split(train_df, test_df, thresh=thresh)
    except Exception as e:
        print(f"[ERROR]: classifier training failed: {e}")
        clf = None
        clf_metrics = {"error": str(e)}
        proba = np.zeros(len(test_df))
        pred = (proba > thresh).astype(int)
        feats = []

    # Stage 2: linear (conditional time/steps)
    try:
        m_steps, m_time, reg_metrics, feats2 = lin.train_two_regressors_split(train_df, test_df)
    except Exception as e:
        print(f"[ERROR]: regressors training failed: {e}")
        m_steps = None
        m_time = None
        reg_metrics = {"error": str(e)}
        feats2 = []

    # Feature lists should match; if not, merge safely
    feats_use = feats if feats == feats2 else list(dict.fromkeys(feats + feats2))

    out = test_df.copy()
    out["p_reach"] = proba
    out["pred_reach"] = pred

    if feats_use and m_steps is not None and m_time is not None:
        missing = [f for f in feats_use if f not in out.columns]
        if missing:
            print(f"[WARNING]: Missing features for prediction: {missing}. Correcting with -1.")
            for f in missing:
                out[f] = -1
        out["pred_steps_to_threshold_cond"] = np.clip(m_steps.predict(out[feats_use]), 0, np.inf)
        out["pred_time_to_threshold_cond"] = np.clip(m_time.predict(out[feats_use]), 0, np.inf)
    else:
        # fallback if models failed
        out["pred_steps_to_threshold_cond"] = 0.0
        out["pred_time_to_threshold_cond"] = 0.0

    if "steps" in out.columns:
        out["expected_steps"] = out["p_reach"] * out["pred_steps_to_threshold_cond"] + (1 - out["p_reach"]) * out[
            "steps"]

    metrics = {"classifier": clf_metrics, "regressors": reg_metrics}
    print("[INFO]: Metrics:")
    print(json.dumps(metrics, indent=2))

    try:
        utils.dump_and_save(ts, out, clf, m_steps, m_time, metrics)
    except Exception as e:
        print(f"[ERROR]: failed to save results: {e}")


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