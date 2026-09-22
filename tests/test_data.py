from satellite_ml.config import FEATURES, TARGET
from satellite_ml.data import load_dataset, split_dataset


def test_dataset_contains_required_columns():
    df = load_dataset()

    assert TARGET in df.columns

    for feature in FEATURES:
        assert feature in df.columns


def test_train_test_split_is_not_empty():
    df = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
