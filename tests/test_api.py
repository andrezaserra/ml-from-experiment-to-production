from fastapi.testclient import TestClient

from satellite_ml.api import app


class DummyModel:
    """Minimal model used to isolate API tests from MLflow."""

    def predict(self, X):
        return [0]


def mock_model_loading(monkeypatch):
    monkeypatch.setattr(
        "satellite_ml.api.mlflow.sklearn.load_model",
        lambda _: DummyModel(),
    )


def test_health(monkeypatch):
    mock_model_loading(monkeypatch)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_predict(monkeypatch):
    mock_model_loading(monkeypatch)

    payload = {
        "battery_voltage": 28.1,
        "battery_current": 1.7,
        "battery_temperature": 24.8,
        "solar_panel_current": 4.9,
        "bus_voltage": 28.0,
        "attitude_error": 0.02,
        "eclipse": 0,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 200

    assert response.json() == {
        "anomaly": False,
        "prediction": 0,
        "model": "satellite-anomaly-classifier",
        "alias": "champion",
    }


def test_predict_rejects_invalid_payload(monkeypatch):
    mock_model_loading(monkeypatch)

    payload = {
        "battery_voltage": 28.1,
        "battery_current": 1.7,
        "battery_temperature": 24.8,
        "solar_panel_current": 4.9,
        "bus_voltage": 28.0,
        "attitude_error": 0.02,
        "eclipse": 3,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422
