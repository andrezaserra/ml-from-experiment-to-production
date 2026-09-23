from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient

from satellite_ml.inference import predict_one
from satellite_ml.schemas import (
    HealthResponse,
    PredictionResponse,
    TelemetryRequest,
)


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "satellite-anomaly-classifier"
MODEL_ALIAS = "champion"
DEFAULT_MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


def resolve_model_metadata(model_uri: str) -> dict:
    """Resolve metadata for the model loaded by the API."""
    manifest_path = os.getenv("MODEL_MANIFEST_PATH")

    if manifest_path:
        path = Path(manifest_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model manifest not found: {path}"
            )

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    if model_uri == DEFAULT_MODEL_URI:
        client = MlflowClient()

        model_version = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS,
        )

        return {
            "model_name": MODEL_NAME,
            "source_alias": MODEL_ALIAS,
            "model_version": str(model_version.version),
            "source_run_id": model_version.run_id,
            "model_uri": model_uri,
        }

    return {
        "model_name": MODEL_NAME,
        "source_alias": None,
        "model_version": "unknown",
        "source_run_id": None,
        "model_uri": model_uri,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction model once when the API starts."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )

    model_uri = os.getenv(
        "MODEL_URI",
        DEFAULT_MODEL_URI,
    )

    mlflow.set_tracking_uri(tracking_uri)

    try:
        app.state.model = mlflow.sklearn.load_model(
            model_uri
        )

        metadata = resolve_model_metadata(
            model_uri
        )

        app.state.model_name = metadata["model_name"]
        app.state.model_version = metadata["model_version"]
        app.state.source_alias = metadata.get(
            "source_alias"
        )

        app.state.model_loaded = True
        app.state.model_load_error = None

        print(f"Model loaded from: {model_uri}")
        print(
            "Model version: "
            f"{app.state.model_version}"
        )

    except Exception as exc:
        app.state.model = None
        app.state.model_loaded = False
        app.state.model_load_error = str(exc)

        print(f"Could not load model: {exc}")

    yield


app = FastAPI(
    title="Satellite Anomaly Detection API",
    description=(
        "API didática para classificação de anomalias "
        "em telemetria sintética."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(request: Request):
    """Return the current health of the prediction service."""
    model_loaded = request.app.state.model_loaded

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    telemetry: TelemetryRequest,
    request: Request,
):
    """Predict whether a telemetry sample is anomalous."""
    if not request.app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available.",
        )

    prediction = predict_one(
        telemetry=telemetry.model_dump(),
        model=request.app.state.model,
    )

    return PredictionResponse(
        anomaly=bool(prediction),
        prediction=int(prediction),
        model=request.app.state.model_name,
        version=request.app.state.model_version,
        source_alias=request.app.state.source_alias,
    )
