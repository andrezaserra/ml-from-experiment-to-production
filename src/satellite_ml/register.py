import argparse
import os

import mlflow
from mlflow import MlflowClient


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "satellite-anomaly-classifier"
LOGGED_MODEL_NAME = "random-forest-model"
CHAMPION_ALIAS = "champion"


def configure_mlflow() -> None:
    """Configure the MLflow tracking URI."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )

    mlflow.set_tracking_uri(tracking_uri)


def find_logged_model(run_id: str):
    """Find the logged model produced by a specific MLflow run."""
    client = MlflowClient()

    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        output_format="list",
    )

    candidates = [
        model
        for model in logged_models
        if model.source_run_id == run_id
        and model.name == LOGGED_MODEL_NAME
    ]

    if not candidates:
        raise RuntimeError(
            f"No logged model named '{LOGGED_MODEL_NAME}' "
            f"was found for run '{run_id}'."
        )

    return max(
        candidates,
        key=lambda model: model.creation_timestamp,
    )


def register_model(
    run_id: str,
    alias: str = CHAMPION_ALIAS,
):
    """Promote a logged model to the MLflow Model Registry."""
    configure_mlflow()

    logged_model = find_logged_model(run_id)

    model_uri = f"models:/{logged_model.model_id}"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=alias,
        version=model_version.version,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="source",
        value="minicourse-v2",
    )

    return model_version, logged_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register an MLflow model version."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID containing the logged model.",
    )

    parser.add_argument(
        "--alias",
        default=CHAMPION_ALIAS,
        help="Alias assigned to the registered model version.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_version, logged_model = register_model(
        run_id=args.run_id,
        alias=args.alias,
    )

    print("Model registered.")
    print(f"Name:       {REGISTERED_MODEL_NAME}")
    print(f"Version:    {model_version.version}")
    print(f"Alias:      {args.alias}")
    print(f"Run ID:     {args.run_id}")
    print(f"Model ID:   {logged_model.model_id}")


if __name__ == "__main__":
    main()