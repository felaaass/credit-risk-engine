from fastapi.testclient import TestClient

from src.api.app import create_app


def _valid_applicant_payload() -> dict[str, float | int]:
    return {
        "limit_bal": 150000.0,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 32,
        "pay_0": 0,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 24500.0,
        "bill_amt2": 23100.0,
        "bill_amt3": 21800.0,
        "bill_amt4": 19400.0,
        "bill_amt5": 18200.0,
        "bill_amt6": 17500.0,
        "pay_amt1": 3500.0,
        "pay_amt2": 3200.0,
        "pay_amt3": 3000.0,
        "pay_amt4": 2500.0,
        "pay_amt5": 2500.0,
        "pay_amt6": 2500.0,
    }


def test_health_endpoint_reports_ready_model() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert isinstance(body["optimal_threshold"], float)


def test_single_prediction_endpoint_returns_risk_and_shap() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=_valid_applicant_payload())
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["probability_of_default"] <= 1.0
        assert data["risk_tier"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        assert data["decision"] in {"APPROVED", "MANUAL_REVIEW", "REJECTED"}
        assert len(data["top_risk_factors"]) == 5


def test_batch_prediction_endpoint_processes_multiple_applicants() -> None:
    app = create_app()
    high_risk_applicant = _valid_applicant_payload()
    high_risk_applicant["pay_0"] = 3
    high_risk_applicant["pay_2"] = 2
    high_risk_applicant["pay_amt1"] = 0.0

    with TestClient(app) as client:
        response = client.post(
            "/v1/predict/batch",
            json={"applicants": [_valid_applicant_payload(), high_risk_applicant]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 2
        assert len(data["predictions"]) == 2
        assert (
            data["predictions"][1]["probability_of_default"]
            > data["predictions"][0]["probability_of_default"]
        )
