from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.preprocess import preprocess_deterministic


DATA_PATH = "data/credit_risk_dataset.csv"
TARGET = "loan_status"

REPORT_PATH = "reports/cv_metrics.csv"

NUM_COLS = [
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
]

CAT_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower_q, axis=0)
        self.hi_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y = y_true[order]

    bad = (y == 1).astype(int)
    good = (y == 0).astype(int)

    n_bad = bad.sum()
    n_good = good.sum()

    if n_bad == 0 or n_good == 0:
        return float("nan")

    cdf_bad = np.cumsum(bad) / n_bad
    cdf_good = np.cumsum(good) / n_good

    return float(np.max(np.abs(cdf_bad - cdf_good)))


def build_pipeline() -> Pipeline:

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ]
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("logreg", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
        )),
    ])

    return model


def main() -> None:

    df = pd.read_csv(DATA_PATH)
    df = preprocess_deterministic(df)

    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET].astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = []

    print("\n===== Cross Validation =====")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = build_pipeline()
        model.fit(X_train, y_train)

        p_valid = model.predict_proba(X_valid)[:, 1]

        roc_auc = roc_auc_score(y_valid, p_valid)
        pr_auc = average_precision_score(y_valid, p_valid)
        ks = ks_statistic(y_valid, p_valid)

        print(f"\nFold {fold}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"KS:      {ks:.4f}")

        metrics.append({
            "fold": fold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "ks": ks,
        })

    metrics_df = pd.DataFrame(metrics)

    print("\n===== CV Summary =====")
    print(metrics_df.describe().loc[["mean", "std"]])

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    metrics_df.to_csv(REPORT_PATH, index=False)

    print("\nSaved report:", REPORT_PATH)


if __name__ == "__main__":
    main()
