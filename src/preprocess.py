from __future__ import annotations
import pandas as pd

TARGET = "loan_status"
DROP_FEATURES = ["person_age", "cb_person_cred_hist_length"]

def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=cols, errors="ignore")

def fix_invalid_emp_length(df: pd.DataFrame, max_years: int = 70) -> pd.DataFrame:
    df = df.copy()
    if "person_emp_length" in df.columns:
        df.loc[df["person_emp_length"] > max_years, "person_emp_length"] = pd.NA
    return df

def preprocess_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Только deterministic логика (без медиан/квантилей).
    """
    df = df.copy()
    df = drop_columns(df, DROP_FEATURES)
    df = fix_invalid_emp_length(df, max_years=70)
    return df

# алиас, чтобы твой make_dataset.py не ломался
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_deterministic(df)
