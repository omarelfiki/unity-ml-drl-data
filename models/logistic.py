import json
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common_features import load_df, make_preprocess, DATA


def train_classifier_split(train_df, test_df, thresh=0.5):
    """Fit on train_df, evaluate + predict on test_df (both already 3DBall)."""
    pre, feats = make_preprocess(train_df)

    Xtr = train_df[feats]
    ytr = train_df["run_reached_threshold"].astype(int).values
    Xte = test_df[feats]
    yte = test_df["run_reached_threshold"].astype(int).values

    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= thresh).astype(int)

    metrics = {
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "features": feats,
        "threshold": float(thresh),
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }
    return clf, metrics, proba, pred, feats
