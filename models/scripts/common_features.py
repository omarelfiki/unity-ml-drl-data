"""
common_features.py

Shared helpers for the prediction models.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ENV = "3DBall"
# Configurations
NUM_FEATS = [
    "steps", "learning_rate", "batch_size", "buffer_size", "epochs", "num_agents", "seed",
    "early_reward_mean", "p_loss_mean", "v_loss_mean", "entropy_mean",
]
CAT_FEATS = ["algorithm"]

def get_newest_data():
    data_dir = "../data/normalized/"
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"[ERROR]: data directory {data_dir} not found")
    available_files = [file for file in p.glob("normalized_v*.csv") if file.is_file()]
    if not available_files:
        raise FileNotFoundError(f"[ERROR]: No data found in {data_dir}")

    def _version_from_path(f: Path) -> int:
        parts = f.stem.rsplit("_v", 1)
        if len(parts) != 2:
            return -1
        try:
            return int(parts[1])
        except ValueError:
            return -1
    newest = max(available_files, key=lambda f: (_version_from_path(f), f.stat().st_mtime))
    return str(newest)

def load_df(path=None, env=ENV):
    if path is None:
        path = get_newest_data()
    print("[INFO]: Loading data from: ", path)
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
