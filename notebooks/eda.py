import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import NUMERIC_FEATURES, TARGET


df = pd.read_csv('data/credit_risk_dataset.csv')

df.info()
df.isna().mean()
df['loan_status'].value_counts(normalize=True)


numeric_df = df[NUMERIC_FEATURES + [TARGET]]

corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

def iqr_outliers_summary(df, cols):
    summary = []
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_share = ((df[col] < lower) | (df[col] > upper)).mean()
        summary.append((col, outlier_share))

    return pd.DataFrame(summary, columns=["feature", "outlier_share"]).sort_values(
        by="outlier_share", ascending=False
    )

outliers_df = iqr_outliers_summary(df, NUMERIC_FEATURES)
print(outliers_df)

print("""В числовых признаках наблюдается умеренная доля выбросов (до ~5%).
Это характерно для финансовых данных.
В дальнейшем для признаков с наибольшей долей выбросов будет применено ограничение экстремальных значений (winsorization), чтобы стабилизировать линейную модель.""")

for col in NUMERIC_FEATURES:
    plt.figure(figsize=(4, 6))
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.show()