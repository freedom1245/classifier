from pathlib import Path
import shutil
import uuid

import torch

from cdc_priority.classifier.baselines import evaluate_baseline_models
from cdc_priority.classifier.evaluate import (
    build_classification_metrics,
    build_confusion_matrix_data,
    export_ablation_results_csv,
)
from cdc_priority.classifier.features import (
    EncodedDataset,
    EncodedSplit,
    FeatureArtifacts,
    NumericStats,
)
from cdc_priority.classifier.model import (
    AttentionTabularClassifier,
    EmbeddingMLPClassifier,
    MLPClassifier,
)
from cdc_priority.classifier.training import _train_prepared_dataset
from cdc_priority.data.dataset_builder import PreparedDataset
from cdc_priority.data.schema import DatasetSchema
from cdc_priority.data.splitter import DatasetSplit


def test_classifier_model_builds() -> None:
    model = MLPClassifier(input_dim=4, hidden_dim=8, num_classes=3, dropout=0.1)
    assert model is not None


def test_embedding_mlp_classifier_builds_and_runs() -> None:
    model = EmbeddingMLPClassifier(
        categorical_vocab_sizes=[4, 8],
        numeric_dim=3,
        hidden_dim=8,
        num_classes=3,
        dropout=0.1,
        embedding_dim=6,
        numeric_projection_dim=5,
    )
    logits = model(
        categorical_inputs=torch.tensor([[1, 2], [0, 3]], dtype=torch.long),
        numeric_inputs=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
    )
    assert logits.shape == (2, 3)


def test_attention_tabular_classifier_builds_and_runs() -> None:
    model = AttentionTabularClassifier(
        categorical_vocab_sizes=[4, 8],
        numeric_dim=3,
        hidden_dim=16,
        num_classes=3,
        dropout=0.1,
        embedding_dim=6,
        attention_dim=12,
        attention_heads=3,
        attention_layers=1,
    )
    logits = model(
        categorical_inputs=torch.tensor([[1, 2], [0, 3]], dtype=torch.long),
        numeric_inputs=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
    )
    assert logits.shape == (2, 3)


def test_build_classification_metrics() -> None:
    metrics = build_classification_metrics(
        labels=[0, 1, 1, 2],
        predictions=[0, 1, 0, 2],
    )

    assert metrics.accuracy == 0.75
    assert metrics.macro_f1 > 0


def test_build_confusion_matrix_data() -> None:
    matrix = build_confusion_matrix_data(
        labels=[0, 1, 1, 2],
        predictions=[0, 1, 0, 2],
        class_names=["high", "low", "medium"],
    )

    assert matrix == [
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]


def test_evaluate_baseline_models() -> None:
    encoded = EncodedDataset(
        train=EncodedSplit(
            categorical=torch.tensor([[0], [1], [0], [1]], dtype=torch.long),
            numeric=torch.tensor(
                [[0.1], [1.0], [0.2], [0.9]],
                dtype=torch.float32,
            ),
            labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        ),
        valid=EncodedSplit(
            categorical=torch.zeros((0, 1), dtype=torch.long),
            numeric=torch.zeros((0, 1), dtype=torch.float32),
            labels=torch.zeros((0,), dtype=torch.long),
        ),
        test=EncodedSplit(
            categorical=torch.tensor([[0], [1]], dtype=torch.long),
            numeric=torch.tensor([[0.15], [0.95]], dtype=torch.float32),
            labels=torch.tensor([0, 1], dtype=torch.long),
        ),
        artifacts=FeatureArtifacts(
            categorical_columns=["event_type"],
            numeric_columns=["business_value"],
            label_encoder=None,
            category_maps={},
            categorical_vocab_sizes=[2],
            numeric_stats=NumericStats(
                means={"business_value": 0.55},
                stds={"business_value": 0.4},
            ),
        ),
    )

    rows = evaluate_baseline_models(
        encoded=encoded,
        class_names=["high", "low"],
        random_state=42,
    )

    model_names = {row["model"] for row in rows}
    assert model_names == {"logistic_regression", "random_forest"}
    assert all("accuracy" in row for row in rows)


def test_export_ablation_results_csv() -> None:
    output_dir = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "classifier"
        / f"pytest-temp-{uuid.uuid4().hex}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_results.csv"
    try:
        export_ablation_results_csv(
            [
                {
                    "variant": "full_model",
                    "removed_columns": "",
                    "accuracy": 0.9,
                    "macro_f1": 0.8,
                    "weighted_f1": 0.88,
                    "high_class_recall": 0.85,
                }
            ],
            output_path,
        )

        content = output_path.read_text(encoding="utf-8")
        assert "variant,removed_columns,accuracy,macro_f1,weighted_f1,high_class_recall" in content
        assert "full_model" in content
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_train_prepared_dataset_smoke() -> None:
    source = {
        "event_type": [
            "view", "cart", "purchase",
            "view", "cart", "purchase",
            "view", "cart", "purchase",
            "view", "cart", "purchase",
        ],
        "table_name": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"],
        "business_domain": [
            "electronics", "electronics", "computers",
            "electronics", "appliances", "computers",
            "electronics", "appliances", "computers",
            "electronics", "electronics", "computers",
        ],
        "source_service": [
            "brand1", "brand2", "brand3",
            "brand1", "brand2", "brand3",
            "brand1", "brand2", "brand3",
            "brand1", "brand2", "brand3",
        ],
        "user_level": [
            "normal", "vip", "internal",
            "normal", "vip", "internal",
            "normal", "vip", "internal",
            "normal", "vip", "internal",
        ],
        "record_size": [10, 12, 15, 11, 13, 16, 12, 14, 17, 13, 15, 18],
        "changed_columns_count": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "estimated_sync_cost": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 1.3, 2.3, 3.3],
        "business_value": [50, 80, 120, 55, 85, 125, 60, 90, 130, 65, 95, 135],
        "consistency_risk": [0.2, 0.4, 0.8, 0.25, 0.45, 0.85, 0.3, 0.5, 0.9, 0.35, 0.55, 0.95],
        "dependency_count": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "queue_wait_time": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "deadline": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "retry_count": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "source_load": [0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4],
        "db_load": [0.2, 0.25, 0.35, 0.2, 0.25, 0.35, 0.2, 0.25, 0.35, 0.2, 0.25, 0.35],
        "is_peak_hour": [0] * 12,
        "is_hot_data": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "priority_label": [
            "low", "medium", "high",
            "low", "medium", "high",
            "low", "medium", "high",
            "low", "medium", "high",
        ],
    }
    import pandas as pd

    full_frame = pd.DataFrame(source)
    schema = DatasetSchema(
        target="priority_label",
        categorical_columns=[
            "event_type",
            "table_name",
            "business_domain",
            "source_service",
            "user_level",
        ],
        numeric_columns=[
            "record_size",
            "changed_columns_count",
            "estimated_sync_cost",
            "business_value",
            "consistency_risk",
            "dependency_count",
            "queue_wait_time",
            "deadline",
            "retry_count",
            "source_load",
            "db_load",
            "is_peak_hour",
            "is_hot_data",
        ],
    )
    prepared = PreparedDataset(
        schema=schema,
        full_frame=full_frame,
        split=DatasetSplit(
            train=full_frame.iloc[:6].copy(),
            valid=full_frame.iloc[6:9].copy(),
            test=full_frame.iloc[9:].copy(),
        ),
        report={"row_count": 12},
    )

    temp_dir = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "classifier"
        / f"pytest-temp-{uuid.uuid4().hex}"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifact = _train_prepared_dataset(
            prepared=prepared,
            config_values={
                "model_variant": "embedding_mlp",
                "batch_size": 4,
                "epochs": 1,
                "hidden_dim": 16,
                "dropout": 0.1,
                "embedding_dim": 8,
                "numeric_projection_dim": 8,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "patience": 1,
                "num_workers": 0,
            },
            output_dir=temp_dir,
            random_state=42,
            run_label="pytest_smoke",
            save_artifacts=False,
        )

        assert artifact["model_variant"] == "embedding_mlp"
        assert "accuracy" in artifact["test"]
        assert "macro_f1" in artifact["validation"]
        assert artifact["classes"] == ["high", "low", "medium"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
