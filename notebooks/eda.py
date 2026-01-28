import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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

df = df.drop(['person_age', 'cb_person_cred_hist_length'], axis=1)
# признаки показали крайне слабую корреляцию т к будет использована линейная регрессия они не нужны 

# заполняем пропуски
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())