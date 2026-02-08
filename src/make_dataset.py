import pandas as pd
from src.preprocess import preprocess



df = pd.read_csv("data/credit_risk_dataset.csv")
df_p = preprocess(df)

df_p.to_csv("data/processed/credit_risk_processed.csv", index=False)
print("Saved:", df_p.shape)
print(df_p.isna().mean().sort_values(ascending=False).head(10))
