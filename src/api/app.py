from __future__ import annotations

import os
import joblib
import pandas as pd

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

MODEL_PATH = "models/logreg.joblib"

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

app = FastAPI(title="Система кредитного скоринга")

_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Модель не найдена: {MODEL_PATH}. Сначала обучи её:\n"
                f"python -m src.models.train_logreg"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def decision_from_proba(proba_default: float, threshold: float = 0.5) -> str:
    return "DECLINE" if proba_default >= threshold else "APPROVE"


def confidence_from_proba(proba_default: float) -> float:
    return abs(proba_default - 0.5) * 2


def decision_ru(decision: str) -> str:
    return "ОТКАЗАТЬ" if decision == "DECLINE" else "ОДОБРИТЬ"


def decision_color(decision: str) -> str:
    return "#ff6b81" if decision == "DECLINE" else "#4ade80"


def percent(v: float) -> str:
    return f"{v * 100:.2f}%"


def render_page(content: str, title: str = "Кредитный скоринг") -> str:
    return f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>
    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      font-family: Inter, Arial, sans-serif;
      min-height: 100%;
      color: #eef2ff;
      background:
        radial-gradient(circle at top left, rgba(91, 33, 182, 0.35), transparent 30%),
        radial-gradient(circle at top right, rgba(59, 130, 246, 0.28), transparent 30%),
        radial-gradient(circle at bottom center, rgba(16, 185, 129, 0.20), transparent 25%),
        linear-gradient(135deg, #071029 0%, #0f172a 45%, #111827 100%);
      background-attachment: fixed;
    }}

    body {{
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 28px;
      overflow-x: hidden;
    }}

    .bg-blur {{
      position: fixed;
      inset: 0;
      pointer-events: none;
      overflow: hidden;
    }}

    .orb {{
      position: absolute;
      border-radius: 50%;
      filter: blur(60px);
      opacity: 0.35;
      animation: float 10s ease-in-out infinite;
    }}

    .orb.one {{
      width: 240px;
      height: 240px;
      background: #7c3aed;
      top: 5%;
      left: 8%;
    }}

    .orb.two {{
      width: 300px;
      height: 300px;
      background: #2563eb;
      right: 8%;
      top: 18%;
      animation-delay: 1.5s;
    }}

    .orb.three {{
      width: 260px;
      height: 260px;
      background: #10b981;
      bottom: 5%;
      left: 32%;
      animation-delay: 3s;
    }}

    @keyframes float {{
      0%, 100% {{
        transform: translateY(0px) translateX(0px) scale(1);
      }}
      50% {{
        transform: translateY(-20px) translateX(10px) scale(1.05);
      }}
    }}

    .shell {{
      width: 100%;
      max-width: 1080px;
      position: relative;
      z-index: 2;
      animation: fadeUp 0.8s ease;
    }}

    @keyframes fadeUp {{
      from {{
        opacity: 0;
        transform: translateY(18px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}

    .card {{
      position: relative;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      box-shadow:
        0 10px 40px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border-radius: 28px;
      padding: 30px;
    }}

    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: center;
      margin-bottom: 26px;
      flex-wrap: wrap;
    }}

    .title {{
      margin: 0;
      font-size: 34px;
      line-height: 1.1;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}

    .subtitle {{
      margin-top: 10px;
      color: #cbd5e1;
      font-size: 15px;
      max-width: 680px;
      line-height: 1.6;
    }}

    .badge {{
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(99, 102, 241, 0.16);
      border: 1px solid rgba(129, 140, 248, 0.32);
      color: #dbeafe;
      font-size: 13px;
      white-space: nowrap;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}

    .field {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .field.full {{
      grid-column: 1 / -1;
    }}

    label {{
      font-size: 14px;
      font-weight: 700;
      color: #f8fafc;
    }}

    input, select {{
      width: 100%;
      border: 1px solid rgba(255, 255, 255, 0.13);
      background: rgba(255, 255, 255, 0.08);
      color: white;
      border-radius: 16px;
      padding: 14px 16px;
      font-size: 15px;
      outline: none;
      transition: 0.25s ease;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }}

    input::placeholder {{
      color: #94a3b8;
    }}

    input:focus, select:focus {{
      border-color: rgba(96, 165, 250, 0.7);
      box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.18);
      transform: translateY(-1px);
    }}

    select option {{
      color: #111827;
    }}

    .hint {{
      color: #94a3b8;
      font-size: 12px;
      line-height: 1.5;
    }}

    .info-panel {{
      margin-top: 22px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}

    .mini-card {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 20px;
      padding: 16px 18px;
    }}

    .mini-label {{
      color: #a5b4fc;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}

    .mini-value {{
      font-size: 24px;
      font-weight: 800;
    }}

    .actions {{
      display: flex;
      gap: 14px;
      margin-top: 26px;
      flex-wrap: wrap;
    }}

    .btn {{
      appearance: none;
      border: 0;
      cursor: pointer;
      padding: 14px 22px;
      border-radius: 16px;
      font-size: 15px;
      font-weight: 800;
      transition: 0.25s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}

    .btn-primary {{
      color: white;
      background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
      box-shadow: 0 10px 24px rgba(59, 130, 246, 0.28);
    }}

    .btn-primary:hover {{
      transform: translateY(-2px);
      box-shadow: 0 14px 30px rgba(59, 130, 246, 0.34);
    }}

    .btn-secondary {{
      color: #e2e8f0;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
    }}

    .btn-secondary:hover {{
      transform: translateY(-2px);
      background: rgba(255,255,255,0.12);
    }}

    .result-wrap {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 22px;
      margin-top: 20px;
    }}

    .decision-box {{
      border-radius: 24px;
      padding: 24px;
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.12);
    }}

    .decision-pill {{
      display: inline-block;
      padding: 10px 16px;
      border-radius: 999px;
      font-weight: 800;
      font-size: 14px;
      letter-spacing: 0.04em;
      margin-bottom: 16px;
    }}

    .kpis {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}

    .kpi {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 18px;
      padding: 16px;
    }}

    .kpi-title {{
      font-size: 13px;
      color: #cbd5e1;
      margin-bottom: 8px;
    }}

    .kpi-value {{
      font-size: 26px;
      font-weight: 800;
      color: white;
    }}

    .payload {{
      margin-top: 18px;
      background: rgba(3, 7, 18, 0.55);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 18px;
      padding: 16px;
      overflow: auto;
      color: #cbd5e1;
      font-size: 13px;
      line-height: 1.55;
    }}

    .footer-note {{
      margin-top: 16px;
      color: #94a3b8;
      font-size: 13px;
    }}

    a {{
      color: #93c5fd;
    }}

    @media (max-width: 860px) {{
      .grid,
      .info-panel,
      .result-wrap {{
        grid-template-columns: 1fr;
      }}

      .title {{
        font-size: 28px;
      }}

      .card {{
        padding: 20px;
        border-radius: 22px;
      }}
    }}
  </style>
</head>
<body>
  <div class="bg-blur">
    <div class="orb one"></div>
    <div class="orb two"></div>
    <div class="orb three"></div>
  </div>

  <div class="shell">
    {content}
  </div>

  <script>
    const incomeInput = document.getElementById("person_income");
    const loanInput = document.getElementById("loan_amnt");
    const ratioValue = document.getElementById("ratioValue");

    function updateRatio() {{
      if (!incomeInput || !loanInput || !ratioValue) return;

      const income = parseFloat(incomeInput.value || "0");
      const loan = parseFloat(loanInput.value || "0");

      if (income > 0 && loan >= 0) {{
        const ratio = loan / income;
        ratioValue.textContent = ratio.toFixed(4);
      }} else {{
        ratioValue.textContent = "—";
      }}
    }}

    if (incomeInput) incomeInput.addEventListener("input", updateRatio);
    if (loanInput) loanInput.addEventListener("input", updateRatio);
    updateRatio();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def crm_form():
    content = """
    <div class="card">
      <div class="hero">
        <div>
          <h1 class="title">Система кредитного скоринга</h1>
          <div class="subtitle">
            Введите параметры клиента и заявки, а модель машинного обучения
            рассчитает вероятность дефолта, уровень уверенности и итоговое решение.
          </div>
        </div>
        <div class="badge">ML + FastAPI + Logistic Regression</div>
      </div>

      <form action="/predict_form" method="post">
        <div class="grid">
          <div class="field">
            <label for="person_income">Доход клиента</label>
            <input id="person_income" name="person_income" type="number" step="0.01" min="0.01" placeholder="Например: 65000" required />
            <div class="hint">Годовой доход клиента.</div>
          </div>

          <div class="field">
            <label for="person_emp_length">Стаж работы (лет)</label>
            <input id="person_emp_length" name="person_emp_length" type="number" step="0.01" min="0" placeholder="Например: 4.5" required />
            <div class="hint">Количество лет на текущей или прошлых работах.</div>
          </div>

          <div class="field">
            <label for="loan_amnt">Сумма кредита</label>
            <input id="loan_amnt" name="loan_amnt" type="number" step="1" min="1" placeholder="Например: 12000" required />
            <div class="hint">Запрашиваемая сумма займа.</div>
          </div>

          <div class="field">
            <label for="loan_int_rate">Процентная ставка (%)</label>
            <input id="loan_int_rate" name="loan_int_rate" type="number" step="0.01" min="0" placeholder="Например: 11.9" required />
            <div class="hint">Годовая процентная ставка по кредиту.</div>
          </div>

          <div class="field">
            <label for="person_home_ownership">Жилищный статус</label>
            <select id="person_home_ownership" name="person_home_ownership" required>
              <option value="RENT">Аренда (RENT)</option>
              <option value="OWN">Собственное жильё (OWN)</option>
              <option value="MORTGAGE">Ипотека (MORTGAGE)</option>
              <option value="OTHER">Другое (OTHER)</option>
            </select>
            <div class="hint">Категориальный признак из обучающего датасета.</div>
          </div>

          <div class="field">
            <label for="loan_intent">Цель кредита</label>
            <select id="loan_intent" name="loan_intent" required>
              <option value="PERSONAL">Личные цели (PERSONAL)</option>
              <option value="EDUCATION">Образование (EDUCATION)</option>
              <option value="MEDICAL">Медицина (MEDICAL)</option>
              <option value="VENTURE">Бизнес / стартап (VENTURE)</option>
              <option value="HOMEIMPROVEMENT">Улучшение жилья (HOMEIMPROVEMENT)</option>
              <option value="DEBTCONSOLIDATION">Рефинансирование долгов (DEBTCONSOLIDATION)</option>
            </select>
            <div class="hint">Используется моделью как один из категориальных факторов.</div>
          </div>

          <div class="field">
            <label for="loan_grade">Класс кредита</label>
            <select id="loan_grade" name="loan_grade" required>
              <option value="A">A</option>
              <option value="B">B</option>
              <option value="C">C</option>
              <option value="D">D</option>
              <option value="E">E</option>
              <option value="F">F</option>
              <option value="G">G</option>
            </select>
            <div class="hint">Чем ниже класс, тем выше риск.</div>
          </div>

          <div class="field">
            <label for="cb_person_default_on_file">Был дефолт в истории</label>
            <select id="cb_person_default_on_file" name="cb_person_default_on_file" required>
              <option value="N">Нет (N)</option>
              <option value="Y">Да (Y)</option>
            </select>
            <div class="hint">Исторический индикатор наличия дефолта.</div>
          </div>

          <div class="field full">
            <label for="threshold">Порог принятия решения</label>
            <input id="threshold" name="threshold" type="number" step="0.01" min="0" max="1" value="0.50" required />
            <div class="hint">
              Если вероятность дефолта выше или равна этому значению, система предложит отказ.
            </div>
          </div>
        </div>

        <div class="info-panel">
          <div class="mini-card">
            <div class="mini-label">Автоматически вычисляется</div>
            <div class="mini-value" id="ratioValue">—</div>
            <div class="hint">Отношение суммы кредита к доходу (loan_percent_income).</div>
          </div>

          <div class="mini-card">
            <div class="mini-label">Документация API</div>
            <div class="mini-value" style="font-size:18px;">
              <a href="/docs" target="_blank">Открыть Swagger UI</a>
            </div>
            <div class="hint">Можно тестировать JSON endpoint прямо из браузера.</div>
          </div>
        </div>

        <div class="actions">
          <button class="btn btn-primary" type="submit">Рассчитать скоринг</button>
          <a class="btn btn-secondary" href="/docs" target="_blank">Открыть API docs</a>
        </div>
      </form>

      <div class="footer-note">
        Интерфейс сделан поверх FastAPI без отдельного фронтенд-фреймворка.
      </div>
    </div>
    """
    return render_page(content, title="Кредитный скоринг")


@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    person_income: float = Form(...),
    person_emp_length: float = Form(...),
    loan_amnt: float = Form(...),
    loan_int_rate: float = Form(...),
    person_home_ownership: str = Form(...),
    loan_intent: str = Form(...),
    loan_grade: str = Form(...),
    cb_person_default_on_file: str = Form(...),
    threshold: float = Form(0.5),
):
    model = load_model()

    income = max(person_income, 1e-9)
    loan_percent_income = loan_amnt / income

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

    result_label = decision_ru(decision)
    color = decision_color(decision)
    approve_text = "Заявка выглядит приемлемой по текущему порогу." if decision == "APPROVE" else "Риск дефолта слишком высокий для текущего порога."

    content = f"""
    <div class="card">
      <div class="hero">
        <div>
          <h1 class="title">Результат скоринга</h1>
          <div class="subtitle">
            Ниже показано решение модели, вероятность дефолта, уровень уверенности
            и вычисленные показатели по введённым данным.
          </div>
        </div>
        <div class="badge">Анализ завершён</div>
      </div>

      <div class="result-wrap">
        <div class="decision-box">
          <div class="decision-pill" style="background:{color}22; color:{color}; border:1px solid {color}55;">
            {result_label}
          </div>

          <div style="font-size:32px; font-weight:900; margin-bottom:10px;">
            {result_label}
          </div>

          <div style="color:#cbd5e1; line-height:1.7; font-size:15px;">
            {approve_text}
          </div>

          <div class="payload">
            <b>Использованные признаки:</b><br/><br/>
            <pre style="margin:0; white-space:pre-wrap;">{payload}</pre>
          </div>
        </div>

        <div class="kpis">
          <div class="kpi">
            <div class="kpi-title">Вероятность дефолта</div>
            <div class="kpi-value">{percent(proba_default)}</div>
          </div>

          <div class="kpi">
            <div class="kpi-title">Уверенность модели</div>
            <div class="kpi-value">{percent(confidence)}</div>
          </div>

          <div class="kpi">
            <div class="kpi-title">Порог решения</div>
            <div class="kpi-value">{threshold:.2f}</div>
          </div>

          <div class="kpi">
            <div class="kpi-title">loan_percent_income</div>
            <div class="kpi-value">{loan_percent_income:.4f}</div>
          </div>
        </div>
      </div>

      <div class="actions">
        <a class="btn btn-primary" href="/">Новая проверка</a>
        <a class="btn btn-secondary" href="/docs" target="_blank">Swagger UI</a>
      </div>
    </div>
    """

    return render_page(content, title="Результат скоринга")


class PredictRequest(BaseModel):
    person_income: float = Field(gt=0)
    person_emp_length: float = Field(ge=0)
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0)
    person_home_ownership: str
    loan_intent: str
    loan_grade: str
    cb_person_default_on_file: str
    threshold: float = Field(default=0.5, ge=0, le=1)


@app.post("/predict_v2", response_class=JSONResponse)
def predict_v2(req: PredictRequest):
    model = load_model()

    loan_percent_income = req.loan_amnt / req.person_income

    payload = {
        "person_income": req.person_income,
        "person_emp_length": req.person_emp_length,
        "loan_amnt": req.loan_amnt,
        "loan_int_rate": req.loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "person_home_ownership": req.person_home_ownership.strip().upper(),
        "loan_intent": req.loan_intent.strip().upper(),
        "loan_grade": req.loan_grade.strip().upper(),
        "cb_person_default_on_file": req.cb_person_default_on_file.strip().upper(),
    }

    X = pd.DataFrame([payload], columns=ALL_COLS)
    proba_default = float(model.predict_proba(X)[:, 1][0])

    decision = decision_from_proba(proba_default, threshold=req.threshold)

    return {
        "decision": decision,
        "decision_ru": decision_ru(decision),
        "proba_default": proba_default,
        "confidence": confidence_from_proba(proba_default),
        "threshold": req.threshold,
        "derived": {"loan_percent_income": loan_percent_income},
    }