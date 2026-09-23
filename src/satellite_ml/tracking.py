import os

import mlflow


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "satellite-anomaly-detection"


def configure_tracking() -> None:
    """Configure the MLflow tracking server and experiment."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
