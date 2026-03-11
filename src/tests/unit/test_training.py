import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def make_dataset():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_ut_06_logistic_regression_reproducibility():
    X, y = make_dataset()

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_1 = LogisticRegression(max_iter=1000, random_state=42)
    model_2 = LogisticRegression(max_iter=1000, random_state=42)

    model_1.fit(X_train_1, y_train_1)
    model_2.fit(X_train_2, y_train_2)

    auc_1 = roc_auc_score(y_test_1, model_1.predict_proba(X_test_1)[:, 1])
    auc_2 = roc_auc_score(y_test_2, model_2.predict_proba(X_test_2)[:, 1])

    assert round(auc_1, 6) == round(auc_2, 6)


def test_ut_07_train_test_split_stratification():
    X, y = make_dataset()

    _, _, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_positive_ratio = y_train.mean()
    test_positive_ratio = y_test.mean()

    assert abs(train_positive_ratio - test_positive_ratio) <= 0.01