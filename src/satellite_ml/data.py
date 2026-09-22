import pandas as pd
from sklearn.model_selection import train_test_split

from satellite_ml.config import (
    DATA_PATH,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Load and validate the telemetry dataset."""
    df = pd.read_csv(path, parse_dates=["timestamp"])

    required_columns = {"timestamp", TARGET, *FEATURES}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    return df


def split_dataset(df: pd.DataFrame):
    """Split features and target into stratified train/test sets."""
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
