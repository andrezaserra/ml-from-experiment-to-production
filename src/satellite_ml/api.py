from contextlib import asynccontextmanager
import os

import mlflow
from fastapi import FastAPI, HTTPException, Request

from satellite_ml.inference import predict_one
from satellite_ml.schemas import (
    HealthResponse,
    PredictionResponse,
    TelemetryRequest,
)


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "satellite-anomaly-classifier"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once when the API starts."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )

    mlflow.set_tracking_uri(tracking_uri)

    try:
        app.state.model = mlflow.sklearn.load_model(
            MODEL_URI
        )
        app.state.model_loaded = True
        app.state.model_load_error = None

        print(
            f"Model loaded: {MODEL_NAME}@{MODEL_ALIAS}"
        )

    except Exception as exc:
        app.state.model = None
        app.state.model_loaded = False
        app.state.model_load_error = str(exc)

        print(
            f"Could not load model: {exc}"
        )

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
        model=MODEL_NAME,
        alias=MODEL_ALIAS,
    )
