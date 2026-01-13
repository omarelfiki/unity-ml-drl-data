import json
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common_features import load_df, make_preprocess, DATA


def train_classifier(df, test_size=0.2, seed=42, thresh=0.5):
    pre, feats = make_preprocess(df)
    X, y = df[feats], df["run_reached_threshold"].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

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
    return clf, metrics


def main():
    df = load_df(DATA)
    clf, metrics = train_classifier(df)

    dump(clf, "logistic_reach_model.joblib")
    with open("logistic_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
