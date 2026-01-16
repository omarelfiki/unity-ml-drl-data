import json
from pathlib import Path
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from scripts.common_features import load_df
import scripts.logistic as log
import scripts.linear as lin
import scripts.utils as utils

def predict(path, data, env, test_size=0.2, seed=42, thresh=0.5, models_dir = None):
    print("[INFO]: Loading dataset splits and grouping features....")
    

    df_all = load_df(path=data, env=None)
    envs = sorted(df_all["environment"].dropna().unique().tolist())
    if env is not None:
        envs = [env]

    for env_name in envs:
        print(f"\n[INFO]: ===== Environment: {env_name} =====")
        df = df_all[df_all["environment"] == env_name].copy()

        # Skip environments that cannot train a classifier (only one class)
        y = df["run_reached_threshold"].astype(int).values
        if len(np.unique(y)) < 2:
            print(f"[SKIP]: {env_name} has only one class in run_reached_threshold.")
            continue
    
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)
        print("[INFO]: Training models....")

        clf = None
        clf_metrics = {}
        reg_metrics = {}
        proba = None
        pred = None
        feats = []
        feats2 = []
        m_steps = None
        m_time = None

        if models_dir:
            print(f"[INFO]: Loading models from {models_dir}....")
            base = Path(models_dir) / str(env_name)
            clf_path = base / "logistic_reach_model.joblib"
            m_steps_path = base / "linear_steps_model.joblib"
            m_time_path = base / "linear_time_model.joblib"

            # classifier
            if clf_path.exists():
                try:
                    clf = load(clf_path)
                    clf_metrics["loaded_from"] = str(clf_path)
                    feat_names = list(getattr(clf, "feature_names_in_", [])) or []
                    if feat_names:
                        try:
                            proba = np.asarray(clf.predict_proba(test_df[feat_names])[:, 1])
                        except Exception:
                            proba = None
                    if proba is None:
                        try:
                            proba = np.asarray(clf.predict_proba(test_df)[:, 1])
                        except Exception:
                            try:
                                proba = np.asarray(clf.predict(test_df))
                            except Exception:
                                proba = None
                    if proba is not None:
                        try:
                            pred = (proba >= thresh).astype(int)
                        except Exception:
                            pred = np.zeros(len(proba), dtype=int)
                    feats = feat_names
                    print(f"[INFO]: Loaded classifier from ` {clf_path} `")
                except Exception as e:
                    print(f"[ERROR]: failed to load classifier from ` {clf_path} `: {e}")
                    clf = None
                    clf_metrics["load_error"] = str(e)

            # regressors
            if m_steps_path.exists():
                try:
                    m_steps = load(m_steps_path)
                    reg_metrics["m_steps_loaded"] = str(m_steps_path)
                    feats2 = list(getattr(m_steps, "feature_names_in_", [])) or feats2
                    print(f"[INFO]: Loaded steps regressor from ` {m_steps_path} `")
                except Exception as e:
                    print(f"[ERROR]: failed to load steps regressor from ` {m_steps_path} `: {e}")
                    reg_metrics["m_steps_error"] = str(e)
                    m_steps = None

            if m_time_path.exists():
                try:
                    m_time = load(m_time_path)
                    reg_metrics["m_time_loaded"] = str(m_time_path)
                    feats2 = list(getattr(m_time, "feature_names_in_", [])) or feats2
                    print(f"[INFO]: Loaded time regressor from ` {m_time_path} `")
                except Exception as e:
                    print(f"[ERROR]: failed to load time regressor from ` {m_time_path} `: {e}")
                    reg_metrics["m_time_error"] = str(e)
                    m_time = None

        if clf is None or proba is None or pred is None:
            print("[INFO]: Training classifier....")
            try:
                clf, clf_metrics, proba, pred, feats = log.train_classifier_split(train_df, test_df, thresh=thresh)
            except Exception as e:
                print(f"[ERROR]: classifier training failed: {e}")
                clf = None
                clf_metrics = {"error": str(e)}
                proba = np.zeros(len(test_df))
                pred = (proba >= thresh).astype(int)
                feats = []

        if m_steps is None or m_time is None:
            print("[INFO]: Training regressors....")
            try:
                m_steps_tr, m_time_tr, reg_metrics_tr, feats2_tr = lin.train_two_regressors_split(train_df, test_df)
                # only set models where absent
                m_steps = m_steps or m_steps_tr
                m_time = m_time or m_time_tr
                # merge metrics and feature lists
                if isinstance(reg_metrics_tr, dict):
                    reg_metrics.update(reg_metrics_tr)
                feats2 = feats2 or feats2_tr
            except Exception as e:
                print(f"[ERROR]: regressors training failed: {e}")
                reg_metrics.update({"error": str(e)})
                m_steps = m_steps or None
                m_time = m_time or None
                feats2 = feats2 or []

        # Feature lists should match; if not matching, then will be merged safely
        feats = feats or []
        feats2 = feats2 or []
        feats_use = feats if feats == feats2 else list(dict.fromkeys(feats + feats2))

        # Normalize proba/pred
        if proba is None:
            proba = np.zeros(len(test_df))
        proba = np.asarray(proba)
        if pred is None:
            try:
                pred = (proba >= thresh).astype(int)
            except Exception:
                pred = np.zeros(len(proba), dtype=int)

        out = test_df.copy()
        out["p_reach"] = proba
        out["pred_reach"] = pred

        if feats_use and m_steps is not None and m_time is not None:
            missing = [f for f in feats_use if f not in out.columns]
            if missing:
                print(f"[WARNING]: Missing features for prediction: {missing}. Correcting with -1.")
                for f in missing:
                    out[f] = np.nan
            try:
                out["pred_steps_to_threshold_cond"] = np.clip(m_steps.predict(out[feats_use]), 0, np.inf)
                out["pred_time_to_threshold_cond"] = np.clip(m_time.predict(out[feats_use]), 0, np.inf)
            except Exception as e:
                print(f"[ERROR]: regressors failed to predict: {e}")
                out["pred_steps_to_threshold_cond"] = 0.0
                out["pred_time_to_threshold_cond"] = 0.0
        else:
            out["pred_steps_to_threshold_cond"] = 0.0
            out["pred_time_to_threshold_cond"] = 0.0

        if "steps" in out.columns:
            out["expected_steps"] = out["p_reach"] * out["pred_steps_to_threshold_cond"] + (1 - out["p_reach"]) * out[
                "steps"]

        metrics = {"classifier": clf_metrics, "regressors": reg_metrics}
        print("[INFO]: Metrics Saved")
        pred_path = Path("experiments") / Path(path) / "predictions"
        pred_path.mkdir(parents=True, exist_ok=True)
        # pred_path = "experiments" / Path(path) / "predictions"
        # pred_path = Path("experiments") / path / "predictions"
        try:
            utils.dump_and_save(pred_path, out, clf, m_steps, m_time, metrics, env_name)
        except Exception as e:
            print(f"[ERROR]: failed to save results: {e}")
