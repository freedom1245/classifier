import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluate import build_classification_metrics, build_classification_report
from .features import EncodedDataset


def build_baseline_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            solver="saga",
            max_iter=5000,
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


def _to_numpy(encoded: EncodedDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_categorical = encoded.train.categorical.cpu().numpy()
    test_categorical = encoded.test.categorical.cpu().numpy()
    train_numeric = encoded.train.numeric.cpu().numpy()
    test_numeric = encoded.test.numeric.cpu().numpy()
    train_labels = encoded.train.labels.cpu().numpy()
    test_labels = encoded.test.labels.cpu().numpy()
    return (
        train_categorical,
        test_categorical,
        train_numeric,
        test_numeric,
        train_labels,
        test_labels,
    )


def build_random_forest_feature_matrices(
    encoded: EncodedDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        train_categorical,
        test_categorical,
        train_numeric,
        test_numeric,
        train_labels,
        test_labels,
    ) = _to_numpy(encoded)
    train_features = _merge_feature_blocks(train_categorical, train_numeric)
    test_features = _merge_feature_blocks(test_categorical, test_numeric)
    return train_features, test_features, train_labels, test_labels


def build_logistic_feature_matrices(
    encoded: EncodedDataset,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    (
        train_categorical,
        test_categorical,
        train_numeric,
        test_numeric,
        train_labels,
        test_labels,
    ) = _to_numpy(encoded)

    feature_blocks_train: list[sparse.spmatrix] = []
    feature_blocks_test: list[sparse.spmatrix] = []

    if train_categorical.size > 0:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        feature_blocks_train.append(encoder.fit_transform(train_categorical))
        feature_blocks_test.append(encoder.transform(test_categorical))

    if train_numeric.size > 0:
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_numeric).astype(np.float32)
        scaled_test = scaler.transform(test_numeric).astype(np.float32)
        feature_blocks_train.append(sparse.csr_matrix(scaled_train))
        feature_blocks_test.append(sparse.csr_matrix(scaled_test))

    if not feature_blocks_train:
        raise ValueError("No features available for logistic regression baseline.")

    train_features = sparse.hstack(feature_blocks_train, format="csr")
    test_features = sparse.hstack(feature_blocks_test, format="csr")
    return train_features, test_features, train_labels, test_labels


def evaluate_baseline_models(
    encoded: EncodedDataset,
    class_names: list[str],
    random_state: int,
    high_class_name: str = "high",
) -> list[dict[str, object]]:
    rf_train_features, rf_test_features, train_labels, test_labels = (
        build_random_forest_feature_matrices(encoded)
    )
    logistic_train_features, logistic_test_features, _, _ = (
        build_logistic_feature_matrices(encoded)
    )
    rows: list[dict[str, object]] = []

    for model_name, model in build_baseline_models(random_state).items():
        if model_name == "logistic_regression":
            train_features = logistic_train_features
            test_features = logistic_test_features
        else:
            train_features = rf_train_features
            test_features = rf_test_features

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
