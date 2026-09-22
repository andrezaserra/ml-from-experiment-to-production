from collections.abc import Mapping

import joblib
import pandas as pd

from satellite_ml.config import FEATURES, MODEL_PATH


def load_model(path=MODEL_PATH):
    """Load the persisted classifier."""
    return joblib.load(path)


def predict_one(
    telemetry: Mapping[str, float],
    model=None,
) -> int:
    """Predict whether one telemetry observation is anomalous."""
    missing = set(FEATURES).difference(telemetry)

    if missing:
        raise ValueError(
            f"Telemetry is missing required features: {sorted(missing)}"
        )

    if model is None:
        model = load_model()

    sample = pd.DataFrame(
        [{feature: telemetry[feature] for feature in FEATURES}]
    )

    prediction = model.predict(sample)[0]

    return int(prediction)
