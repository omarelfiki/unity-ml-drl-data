"""
common_features.py

Shared helpers for the 3DBall prediction models.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ENV = "3DBall"
DATA = "../data/prediction_snapshot.csv"

# Configurations
NUM_FEATS = [
    "steps","learning_rate","batch_size","buffer_size","epochs","num_agents","seed",
    "early_reward_mean","p_loss_mean","v_loss_mean","entropy_mean",
]
CAT_FEATS = ["algorithm"]


def load_df(path=DATA, env=ENV):
    df = pd.read_csv(path)
    for c in ["steps_to_threshold", "time_to_threshold"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["run_reached_threshold"] = ((df["steps_to_threshold"] > 0) & (df["time_to_threshold"] > 0)).astype(int)
    return df[df["environment"] == env].copy()


def make_preprocess(df, num_feats=NUM_FEATS, cat_feats=CAT_FEATS):
    num = [c for c in num_feats if c in df.columns]
    cat = [c for c in cat_feats if c in df.columns]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
    )
    return pre, (num + cat)
