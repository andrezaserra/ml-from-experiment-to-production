from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(y_true, y_pred) -> dict:
    """Calculate the baseline classification metrics."""
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }
