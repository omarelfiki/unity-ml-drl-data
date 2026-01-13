import json
import numpy as np
from joblib import dump
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common_features import load_df, make_preprocess, DATA


def train_two_regressors_split(train_df, test_df):
    """Fit on reached-only train_df; evaluate on reached-only test_df."""
    tr = train_df[(train_df["run_reached_threshold"] == 1) &
                  (train_df["steps_to_threshold"] > 0) &
                  (train_df["time_to_threshold"] > 0)].copy()

    te = test_df[(test_df["run_reached_threshold"] == 1) &
                 (test_df["steps_to_threshold"] > 0) &
                 (test_df["time_to_threshold"] > 0)].copy()

    if len(tr) < 5:
        raise ValueError("Not enough reached runs in training split to fit regressors.")

    pre, feats = make_preprocess(tr)

    base = Pipeline([("pre", pre), ("lr", LinearRegression())])
    ytfm = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
    mk = lambda: TransformedTargetRegressor(regressor=base, transformer=ytfm, check_inverse=False)

    m_steps, m_time = mk(), mk()
    m_steps.fit(tr[feats], tr["steps_to_threshold"].values)
    m_time.fit(tr[feats], tr["time_to_threshold"].values)

    metrics = {
        "n_train_reached": int(len(tr)),
        "n_test_reached": int(len(te)),
        "features": feats,
    }

    if len(te) >= 1:
        ps = np.clip(m_steps.predict(te[feats]), 0, np.inf)
        pt = np.clip(m_time.predict(te[feats]), 0, np.inf)

        metrics["steps"] = {
            "mae": float(mean_absolute_error(te["steps_to_threshold"], ps)),
            "rmse": float(np.sqrt(mean_squared_error(te["steps_to_threshold"], ps))),
            "r2": float(r2_score(te["steps_to_threshold"], ps)) if len(te) >= 2 else float("nan"),
        }
        metrics["time"] = {
            "mae": float(mean_absolute_error(te["time_to_threshold"], pt)),
            "rmse": float(np.sqrt(mean_squared_error(te["time_to_threshold"], pt))),
            "r2": float(r2_score(te["time_to_threshold"], pt)) if len(te) >= 2 else float("nan"),
        }

    return m_steps, m_time, metrics, feats
