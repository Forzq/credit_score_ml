from __future__ import annotations

import os
import pandas as pd
from src.preprocess import preprocess


RAW_PATH = "data/credit_risk_dataset.csv"
OUT_PATH = "data/processed/credit_risk_processed.csv"


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    df_p = preprocess(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_p.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH, df_p.shape)
    print("Top NA columns:")
    print(df_p.isna().mean().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
