from __future__ import annotations
import pandas as pd


NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET = "loan_status"

# удаляем слабокоррелируемые столбцы + у них пропуски
DROP_FEATURES = ["person_age", "cb_person_cred_hist_length"]

# клиппинг выбросов, которые в этом нуждаются
OUTLIER_CLIP_COLS = ["loan_amnt", "person_income"]


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=cols, errors="ignore")

# исправляем логическую проблему со стажем свыше 70 лет
def fix_invalid_emp_length(df: pd.DataFrame, max_years: int = 70) -> pd.DataFrame:

    df = df.copy()

    if "person_emp_length" in df.columns:
        df.loc[df["person_emp_length"] > max_years, "person_emp_length"] = pd.NA

    return df



def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "loan_int_rate" in df.columns:
        df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    if "person_emp_length" in df.columns:
        df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    return df


def clip_outliers(df: pd.DataFrame, cols: list[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        df[col] = df[col].clip(lo, hi)
    return df


def preprocess(df: pd.DataFrame, do_clip_outliers: bool = True) -> pd.DataFrame:
    df = df.copy()

    df = drop_columns(df, DROP_FEATURES)

    df = fix_invalid_emp_length(df, max_years=70)

    df = fill_missing_values(df)

    if do_clip_outliers:
        df = clip_outliers(df, OUTLIER_CLIP_COLS)

    return df
