import argparse

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier

from satellite_ml.config import (
    RANDOM_STATE,
)
from satellite_ml.data import load_dataset, split_dataset
from satellite_ml.evaluate import evaluate_classifier
from satellite_ml.tracking import configure_tracking


DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 10

SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
]


def build_model(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> RandomForestClassifier:
    """Create the Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def train(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    run_name: str | None = None,
):
    """Train, evaluate, persist and track the model."""
    configure_tracking()

    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run(run_name=run_name) as run:
        model = build_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = evaluate_classifier(
            y_test,
            predictions,
        )

        # Parameters describe how the model was configured.
        mlflow.log_params(params)

        # Metrics describe the result of the run.
        mlflow.log_metrics(
            {
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        # Non-scalar results can be stored as artifacts.
        mlflow.log_dict(
            metrics,
            "metrics/metrics.json",
        )

        # Store metadata that helps identify this run.
        mlflow.set_tags(
            {
                "project": "ml-from-experiment-to-production",
                "stage": "v2-mlflow",
                "dataset": "synthetic-satellite-telemetry",
            }
        )

        # Describe the expected inputs and outputs of the model.
        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        # Log the model using the MLflow sklearn flavor.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random-forest-model",
            signature=signature,
            input_example=X_train.head(5),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        return model, metrics, run.info.run_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the satellite anomaly classifier."
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    _, metrics, run_id = train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        run_name=args.run_name,
    )

    print("Training completed.")
    print(f"Run ID:     {run_id}")
    print(f"F1:         {metrics['f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()