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
