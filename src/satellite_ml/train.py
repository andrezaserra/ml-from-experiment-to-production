import json

import joblib
from sklearn.ensemble import RandomForestClassifier

from satellite_ml.config import (
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
)
from satellite_ml.data import load_dataset, split_dataset
from satellite_ml.evaluate import evaluate_classifier


def build_model() -> RandomForestClassifier:
    """Create the baseline classifier."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def train():
    """Train, evaluate and persist the baseline model."""
    df = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(df)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = evaluate_classifier(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return model, metrics


def main():
    _, metrics = train()

    print("Training completed.")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Model:     {MODEL_PATH}")
    print(f"Metrics:   {METRICS_PATH}")


if __name__ == "__main__":
    main()
