import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .evaluate import build_classification_metrics, build_classification_report
from .features import EncodedDataset


def build_baseline_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
    }


def _merge_feature_blocks(categorical: np.ndarray, numeric: np.ndarray) -> np.ndarray:
    if categorical.size == 0:
        return numeric
    if numeric.size == 0:
        return categorical.astype(np.float32)
    return np.concatenate([categorical.astype(np.float32), numeric], axis=1)


def build_sklearn_feature_matrices(
    encoded: EncodedDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features = _merge_feature_blocks(
        encoded.train.categorical.cpu().numpy(),
        encoded.train.numeric.cpu().numpy(),
    )
    test_features = _merge_feature_blocks(
        encoded.test.categorical.cpu().numpy(),
        encoded.test.numeric.cpu().numpy(),
    )
    train_labels = encoded.train.labels.cpu().numpy()
    test_labels = encoded.test.labels.cpu().numpy()
    return train_features, test_features, train_labels, test_labels


def evaluate_baseline_models(
    encoded: EncodedDataset,
    class_names: list[str],
    random_state: int,
    high_class_name: str = "high",
) -> list[dict[str, object]]:
    train_features, test_features, train_labels, test_labels = build_sklearn_feature_matrices(encoded)
    rows: list[dict[str, object]] = []

    for model_name, model in build_baseline_models(random_state).items():
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        metrics = build_classification_metrics(test_labels.tolist(), predictions.tolist())
        report = build_classification_report(
            test_labels.tolist(),
            predictions.tolist(),
            class_names,
        )
        high_class_recall = float(report.get(high_class_name, {}).get("recall", 0.0))

        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "high_class_recall": high_class_recall,
            }
        )

    return rows
