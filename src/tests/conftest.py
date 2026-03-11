import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def valid_payload():
    return {
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