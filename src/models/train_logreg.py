from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.preprocess import preprocess_deterministic


DATA_PATH = "data/processed/credit_risk_processed.csv"
TARGET = "loan_status"
MODEL_PATH = "models/logreg.joblib"

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


from src.features.clipper import QuantileClipper



# ===== KS Metric =====
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


def main() -> None:

    # 1️⃣ Load
    df = pd.read_csv(DATA_PATH)

    # 2️⃣ Deterministic preprocess
    df = preprocess_deterministic(df)

    # 3️⃣ X / y
    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET].astype(int).values

    # 4️⃣ Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5️⃣ Numeric pipeline
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])

    # 6️⃣ Categorical pipeline
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    # 7️⃣ ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ]
    )

    # 8️⃣ Final model pipeline
    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("logreg", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
        )),
    ])

    # 9️⃣ Fit
    model.fit(X_train, y_train)

    # 🔟 Predict
    p_test = model.predict_proba(X_test)[:, 1]

    # 1️⃣1️⃣ Metrics
    roc_auc = roc_auc_score(y_test, p_test)
    pr_auc = average_precision_score(y_test, p_test)
    ks = ks_statistic(y_test, p_test)

    print("\n===== Logistic Regression Baseline =====")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"KS:      {ks:.4f}")

    # 1️⃣2️⃣ Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nSaved:", MODEL_PATH)


if __name__ == "__main__":
    main()
