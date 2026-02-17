# credit_score_ml
# How to start project:
# python -m venv venv - create virtual environment
# venv\Scripts\activate - activate env
# pip install -r requirements.txt - set dependencies
# uvicorn src.api.app:app --reload - start api

# Logistic Regression baseline with preprocessing pipeline
# 5-fold CV results: ROC-AUC = 0.87 ± 0.003, KS = 0.61 ± 0.004

# future goals:
# add gradient boosting model (LightGBM / XGBoost) and compare with Logistic Regression baseline
# dockerize the API service
