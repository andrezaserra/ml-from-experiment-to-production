import pytest
from fastapi.testclient import TestClient

from satellite_ml.api import app


class DummyModel:
    """Minimal model used to isolate API tests from model infrastructure."""

    def predict(self, X):
        return [0]


@pytest.fixture
def client():
    """
    Configure the application state directly.

    The TestClient is intentionally not used as a context manager,
    so the application lifespan is not executed during these unit tests.
    """
    app.state.model = DummyModel()
    app.state.model_loaded = True
    app.state.model_load_error = None

    app.state.model_name = "satellite-anomaly-classifier"
    app.state.model_version = "1"
    app.state.source_alias = "champion"

    test_client = TestClient(app)

    yield test_client

    test_client.close()


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_predict(client):
    payload = {
        "battery_voltage": 28.1,
        "battery_current": 1.7,
        "battery_temperature": 24.8,
        "solar_panel_current": 4.9,
        "bus_voltage": 28.0,
        "attitude_error": 0.02,
        "eclipse": 0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    assert response.json() == {
        "anomaly": False,
        "prediction": 0,
        "model": "satellite-anomaly-classifier",
        "version": "1",
        "source_alias": "champion",
    }


def test_predict_rejects_invalid_payload(client):
    payload = {
        "battery_voltage": 28.1,
        "battery_current": 1.7,
        "battery_temperature": 24.8,
        "solar_panel_current": 4.9,
        "bus_voltage": 28.0,
        "attitude_error": 0.02,
        "eclipse": 3,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
