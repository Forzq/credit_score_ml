from locust import HttpUser, between, task


PREDICT_PAYLOAD = {
    "person_income": 65000,
    "person_emp_length": 4,
    "loan_amnt": 12000,
    "loan_int_rate": 11.5,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "cb_person_default_on_file": "N",
    "threshold": 0.5,
}


class CreditScoringUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(9)
    def predict(self):
        self.client.post("/predict", json=PREDICT_PAYLOAD, name="POST /predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")