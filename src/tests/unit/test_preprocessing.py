import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


class TrainOnlyPipeline(Pipeline):
    def fit_transform(self, X, y=None, **fit_params):
        if hasattr(X, "attrs") and X.attrs.get("is_test", False):
            raise AssertionError("fit_transform on test is forbidden")
        return super().fit_transform(X, y=y, **fit_params)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "person_income": [50000, 60000, np.nan, 80000, 90000],
            "person_emp_length": [1, 2, 3, np.nan, 5],
            "loan_amnt": [10000, 12000, 15000, 18000, np.nan],
            "loan_int_rate": [10.5, 11.2, np.nan, 13.0, 9.8],
            "loan_percent_income": [0.2, 0.2, 0.25, np.nan, 0.1],
            "person_home_ownership": ["RENT", "OWN", None, "MORTGAGE", "RENT"],
            "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", None, "VENTURE"],
            "loan_grade": ["A", "B", None, "C", "D"],
            "cb_person_default_on_file": ["N", None, "Y", "N", "N"],
        }
    )


@pytest.fixture
def train_test_df(sample_df):
    train = sample_df.iloc[:3].copy()
    test = sample_df.iloc[3:].copy()
    test.attrs["is_test"] = True
    return train, test


def make_preprocessor():
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )


def test_ut_01_simple_imputer_numeric_median(train_test_df):
    train, _ = train_test_df
    preprocessor = make_preprocessor()

    preprocessor.fit(train)

    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    medians = num_imputer.statistics_

    expected_median_income = np.nanmedian(train["person_income"].to_numpy())
    actual_median_income = medians[NUM_COLS.index("person_income")]

    assert not np.isnan(medians).any()
    assert actual_median_income == expected_median_income

    transformed_train = preprocessor.transform(train)
    transformed_train = np.asarray(transformed_train)

    assert not np.isnan(transformed_train[:, : len(NUM_COLS)]).any()


def test_ut_02_simple_imputer_categorical_missing(sample_df):
    preprocessor = make_preprocessor()
    preprocessor.fit(sample_df)

    cat_pipeline = preprocessor.named_transformers_["cat"]
    cat_imputer = cat_pipeline.named_steps["imputer"]

    cat_imputed = cat_imputer.transform(sample_df[CAT_COLS])

    assert pd.isna(cat_imputed).sum() == 0
    assert "missing" in cat_imputed


def test_ut_03_standard_scaler(sample_df):
    preprocessor = make_preprocessor()
    preprocessor.fit(sample_df)

    num_pipeline = preprocessor.named_transformers_["num"]
    imputed = num_pipeline.named_steps["imputer"].transform(sample_df[NUM_COLS])
    scaled = num_pipeline.named_steps["scaler"].transform(imputed)

    means = scaled.mean(axis=0)
    stds = scaled.std(axis=0, ddof=0)

    assert np.allclose(means, 0, atol=1e-7)
    assert np.allclose(stds, 1, atol=1e-7)


def test_ut_04_pipeline_anti_leakage(train_test_df):
    train, test = train_test_df
    preprocessor = make_preprocessor()
    pipe = TrainOnlyPipeline([("preprocessor", preprocessor)])

    pipe.fit_transform(train)

    with pytest.raises(AssertionError):
        pipe.fit_transform(test)


def test_ut_05_pipeline_serialization(sample_df, tmp_path):
    preprocessor = make_preprocessor()
    preprocessor.fit(sample_df)

    original = preprocessor.transform(sample_df)
    original = np.asarray(original)

    pkl_path = tmp_path / "pipeline.pkl"
    joblib.dump(preprocessor, pkl_path)

    loaded = joblib.load(pkl_path)
    restored = loaded.transform(sample_df)
    restored = np.asarray(restored)

    assert np.allclose(original, restored)