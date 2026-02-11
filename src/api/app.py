from __future__ import annotations

import os
import joblib
import pandas as pd

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

MODEL_PATH = "models/logreg.joblib"

# Те же поля, что использовались в обучении
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

ALL_COLS = NUM_COLS + CAT_COLS

app = FastAPI(title="Credit Scoring CRM (LogReg)")

_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Не найден файл модели: {MODEL_PATH}. Сначала обучи модель:\n"
                f"python -m src.models.train_logreg"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def decision_from_proba(proba_default: float, threshold: float = 0.5) -> str:
    # threshold можно потом заменить на “оптимальный по KS”
    return "DECLINE" if proba_default >= threshold else "APPROVE"


def confidence_from_proba(proba_default: float) -> float:
    # “уверенность” как расстояние от 0.5
    # 0.50 -> 0% уверенности, 0.90 -> 80% уверенности (в APPROVE/DECLINE)
    return abs(proba_default - 0.5) * 2


@app.get("/", response_class=HTMLResponse)
def crm_form():
    # Очень простой “CRM”-вид: форма + подсказки по категориям
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Credit Scoring CRM</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 860px; margin: 30px auto; }
    .card { padding: 18px; border: 1px solid #ddd; border-radius: 14px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    label { font-weight: 600; }
    input { width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #ccc; }
    .btn { margin-top: 14px; padding: 12px 16px; border-radius: 12px; border: 0; cursor: pointer; }
    .hint { color: #555; font-size: 12px; margin-top: 4px; }
    h2 { margin-top: 0; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Credit Scoring CRM (ಠ_ಠ)</h2>
    <form action="/predict_form" method="post">
      <div class="row">
        <div>
          <label>person_income</label>
          <input name="person_income" type="number" step="0.01" required/>
          <div class="hint">Напр. 59000</div>
        </div>
        <div>
          <label>person_emp_length</label>
          <input name="person_emp_length" type="number" step="0.01" required/>
          <div class="hint">Стаж (лет), напр. 4</div>
        </div>

        <div>
          <label>loan_amnt</label>
          <input name="loan_amnt" type="number" step="1" required/>
          <div class="hint">Сумма кредита</div>
        </div>
        <div>
          <label>loan_int_rate</label>
          <input name="loan_int_rate" type="number" step="0.01" required/>
          <div class="hint">Процентная ставка, напр. 16.02</div>
        </div>

        <div>
          <label>loan_percent_income</label>
          <input name="loan_percent_income" type="number" step="0.01" required/>
          <div class="hint">Доля платежа от дохода, напр. 0.59</div>
        </div>
        <div>
          <label>person_home_ownership</label>
          <input name="person_home_ownership" type="text" required/>
          <div class="hint">RENT / OWN / MORTGAGE / OTHER</div>
        </div>

        <div>
          <label>loan_intent</label>
          <input name="loan_intent" type="text" required/>
          <div class="hint">PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION</div>
        </div>
        <div>
          <label>loan_grade</label>
          <input name="loan_grade" type="text" required/>
          <div class="hint">A / B / C / D / E / F / G</div>
        </div>

        <div>
          <label>cb_person_default_on_file</label>
          <input name="cb_person_default_on_file" type="text" required/>
          <div class="hint">Y / N</div>
        </div>

        <div>
          <label>Threshold</label>
          <input name="threshold" type="number" step="0.01" value="0.50" required/>
          <div class="hint">Порог дефолта для DECLINE</div>
        </div>
      </div>

      <button class="btn" type="submit">Скоринг</button>
    </form>

    <p style="margin-top: 16px;">
      API docs: <a href="/docs">/docs</a>
    </p>
  </div>
</body>
</html>
"""


@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    person_income: float = Form(...),
    person_emp_length: float = Form(...),
    loan_amnt: float = Form(...),
    loan_int_rate: float = Form(...),
    loan_percent_income: float = Form(...),
    person_home_ownership: str = Form(...),
    loan_intent: str = Form(...),
    loan_grade: str = Form(...),
    cb_person_default_on_file: str = Form(...),
    threshold: float = Form(0.5),
):
    model = load_model()

    payload = {
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "person_home_ownership": person_home_ownership.strip().upper(),
        "loan_intent": loan_intent.strip().upper(),
        "loan_grade": loan_grade.strip().upper(),
        "cb_person_default_on_file": cb_person_default_on_file.strip().upper(),
    }

    X = pd.DataFrame([payload], columns=ALL_COLS)

    proba_default = float(model.predict_proba(X)[:, 1][0])
    decision = decision_from_proba(proba_default, threshold=threshold)
    confidence = confidence_from_proba(proba_default)

    # “процент уверенности” = confidence * 100
    return f"""
<!doctype html>
<html><head><meta charset="utf-8"/><title>Result</title></head>
<body style="font-family: Arial; max-width: 860px; margin: 30px auto;">
  <div style="padding:18px;border:1px solid #ddd;border-radius:14px;">
    <h2>Result (ಠ_ಠ)</h2>
    <p><b>Decision:</b> {decision}</p>
    <p><b>Probability of default:</b> {proba_default*100:.2f}%</p>
    <p><b>Confidence:</b> {confidence*100:.2f}%</p>
    <p><b>Threshold:</b> {threshold:.2f}</p>
    <hr/>
    <pre>{payload}</pre>
    <p><a href="/">← back</a></p>
  </div>
</body></html>
"""


@app.post("/predict", response_class=JSONResponse)
def predict_json(payload: dict):
    """
    Для интеграции с “настоящим” фронтом.
    Возвращает decision + вероятность дефолта + уверенность.
    """
    model = load_model()
    X = pd.DataFrame([payload], columns=ALL_COLS)

    proba_default = float(model.predict_proba(X)[:, 1][0])
    threshold = float(payload.get("threshold", 0.5))

    return {
        "decision": decision_from_proba(proba_default, threshold=threshold),
        "proba_default": proba_default,
        "confidence": confidence_from_proba(proba_default),
        "threshold": threshold,
    }
