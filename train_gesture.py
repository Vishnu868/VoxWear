"""Train the XGBoost gesture classifier on landmarks stored in dataset/metadata.csv."""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from voxwear.gesture import landmark_features


def augment(lm: np.ndarray, rng, n: int = 3):
    """Small rotations (±5°), scale and jitter in landmark space."""
    out = []
    for _ in range(n):
        a = np.deg2rad(rng.uniform(-5, 5))
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        c = lm[0]
        out.append((lm - c) @ R.T * rng.uniform(0.95, 1.05) + c + rng.normal(0, 0.003, lm.shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="models/gesture_xgb.joblib")
    ap.add_argument("--augment", type=int, default=3)
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.data, "metadata.csv"))
    lms = [np.array(json.loads(s)) for s in meta["landmarks"]]
    enc = LabelEncoder().fit(meta["gesture"])
    y = enc.transform(meta["gesture"])

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    rng = np.random.default_rng(42)
    X_tr, y_tr = [], []
    for i in idx_tr:                                   # augment training data only
        for lm in [lms[i]] + augment(lms[i], rng, args.augment):
            X_tr.append(landmark_features(lm))
            y_tr.append(y[i])
    X_te = np.array([landmark_features(lms[i]) for i in idx_te])
    y_te = y[idx_te]

    model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                          colsample_bytree=0.8, eval_metric="mlogloss")
    model.fit(np.array(X_tr), np.array(y_tr))
    pred = model.predict(X_te)
    print(f"Classes: {len(enc.classes_)} | train {len(X_tr)} | test {len(X_te)}")
    print(f"Test accuracy: {accuracy_score(y_te, pred):.4f} | "
          f"macro F1: {f1_score(y_te, pred, average='macro'):.4f}")
    print(classification_report(y_te, pred, target_names=enc.classes_, zero_division=0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": model, "labels": list(enc.classes_)}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
