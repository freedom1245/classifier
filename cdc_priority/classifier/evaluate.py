from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float


def build_classification_metrics(
    labels: list[int],
    predictions: list[int],
) -> ClassificationMetrics:
    total = max(len(labels), 1)
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / total
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    )


def build_classification_report(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict[str, object]:
    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )


def build_confusion_matrix_data(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> list[list[int]]:
    label_ids = list(range(len(class_names)))
    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    return matrix.tolist()


def export_confusion_matrix_figure(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: Path,
    title: str = "Classifier Confusion Matrix",
) -> Path:
    matrix = build_confusion_matrix_data(labels, predictions, class_names)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            axis.text(col_index, row_index, str(value), ha="center", va="center")

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def export_baseline_comparison_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "high_class_recall",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def export_ablation_results_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "removed_columns",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "high_class_recall",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
