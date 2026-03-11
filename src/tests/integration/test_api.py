import pytest

import src.api.app as app_module


@pytest.mark.asyncio
async def test_it_01_post_predict_valid_request(async_client, valid_payload):
    response = await async_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    data = response.json()

    assert 0 <= data["probability"] <= 1
    assert data["risk_class"] in {"low", "medium", "high"}


@pytest.mark.asyncio
async def test_it_02_post_predict_missing_required_field(async_client, valid_payload):
    payload = valid_payload.copy()
    payload.pop("loan_amnt")

    response = await async_client.post("/predict", json=payload)

    assert response.status_code == 422
    data = response.json()

    assert any(err["loc"][-1] == "loan_amnt" for err in data["detail"])
    assert any("Field required" in err["msg"] or "field required" in err["msg"] for err in data["detail"])


@pytest.mark.asyncio
async def test_it_03_post_predict_invalid_field_type(async_client, valid_payload):
    payload = valid_payload.copy()
    payload["person_income"] = "not_a_number"

    response = await async_client.post("/predict", json=payload)

    assert response.status_code == 422
    data = response.json()

    assert len(data["detail"]) > 0


@pytest.mark.asyncio
async def test_it_04_post_predict_boundary_values(async_client, valid_payload):
    payload = valid_payload.copy()
    payload.update(
        {
            "person_income": 1,
            "person_emp_length": 0,
            "loan_amnt": 999999999,
            "loan_int_rate": 0,
        }
    )

    response = await async_client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert 0 <= data["probability"] <= 1
    assert data["risk_class"] in {"low", "medium", "high"}


@pytest.mark.asyncio
async def test_it_05_post_predict_empty_json(async_client):
    response = await async_client.post("/predict", json={})

    assert response.status_code == 422
    data = response.json()

    required_fields = {
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    }

    actual_fields = {err["loc"][-1] for err in data["detail"]}
    assert required_fields.issubset(actual_fields)


@pytest.mark.asyncio
async def test_it_06_get_health_ok(async_client):
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "model_version" in data
    assert "loaded_at" in data


@pytest.mark.asyncio
async def test_it_07_get_health_model_missing(async_client, monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "MODEL_PATH", "models/definitely_missing_model.joblib")

    response = await async_client.get("/health")

    assert response.status_code == 503
    data = response.json()

    assert "detail" in data


@pytest.mark.asyncio
async def test_it_08_get_metrics_after_predict_calls(async_client, valid_payload):
    for _ in range(3):
        response = await async_client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    response = await async_client.get("/metrics")

    assert response.status_code == 200
    data = response.json()

    assert data["predict_requests_total"] > 0
    assert "histogram_latency_ms" in data