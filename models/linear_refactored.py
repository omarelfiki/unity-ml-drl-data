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


def train_two_regressors(df, test_size=0.2, seed=42):
    # Stage-2: train only on reached runs (targets > 0)
    df = df[(df["run_reached_threshold"] == 1) &
            (df["steps_to_threshold"] > 0) & (df["time_to_threshold"] > 0)].copy()

    pre, feats = make_preprocess(df)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    Xtr, Xte = train_df[feats], test_df[feats]

    base = Pipeline([("pre", pre), ("lr", LinearRegression())])
    ytfm = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
    make = lambda: TransformedTargetRegressor(regressor=base, transformer=ytfm, check_inverse=False)

    m_steps, m_time = make(), make()
    m_steps.fit(Xtr, train_df["steps_to_threshold"].values)
    m_time.fit(Xtr, train_df["time_to_threshold"].values)

    p_steps = np.clip(m_steps.predict(Xte), 0, np.inf)
    p_time = np.clip(m_time.predict(Xte), 0, np.inf)

    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features": feats,
        "steps": {
            "mae": float(mean_absolute_error(test_df["steps_to_threshold"], p_steps)),
            "rmse": float(np.sqrt(mean_squared_error(test_df["steps_to_threshold"], p_steps))),
            "r2": float(r2_score(test_df["steps_to_threshold"], p_steps)) if len(test_df) >= 2 else float("nan"),
        },
        "time": {
            "mae": float(mean_absolute_error(test_df["time_to_threshold"], p_time)),
            "rmse": float(np.sqrt(mean_squared_error(test_df["time_to_threshold"], p_time))),
            "r2": float(r2_score(test_df["time_to_threshold"], p_time)) if len(test_df) >= 2 else float("nan"),
        },
    }
    return m_steps, m_time, metrics


def main():
    df = load_df(DATA)
    m_steps, m_time, metrics = train_two_regressors(df)

    dump(m_steps, "linear_steps_model.joblib")
    dump(m_time, "linear_time_model.joblib")
    with open("linear_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
