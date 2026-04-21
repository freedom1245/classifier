import json
import random

from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import EncodedData
from .model import TabularTraceDataset, ThesiosClassifier, ThesiosClassifierV2


def build_model(args, encoded: EncodedData) -> nn.Module:
    model_class = {
        "v1": ThesiosClassifier,
        "v2": ThesiosClassifierV2,
    }[args.model_variant]
    return model_class(
        categorical_vocab_sizes=encoded.categorical_vocab_sizes,
        numeric_dim=encoded.train_numeric.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=len(encoded.label_encoder.classes_),
        dropout=args.dropout,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for categorical_inputs, numeric_inputs, labels in data_loader:
            categorical_inputs = categorical_inputs.to(device)
            numeric_inputs = numeric_inputs.to(device)
            labels = labels.to(device)

            logits = model(categorical_inputs, numeric_inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = (
        sum(int(pred == label) for pred, label in zip(all_predictions, all_labels))
        / max(total_examples, 1)
    )
    average_loss = total_loss / max(total_examples, 1)
    return average_loss, accuracy, all_labels, all_predictions


def train_and_save(args, encoded: EncodedData) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TabularTraceDataset(
        encoded.train_categorical,
        encoded.train_numeric,
        encoded.train_labels,
    )
    valid_dataset = TabularTraceDataset(
        encoded.valid_categorical,
        encoded.valid_numeric,
        encoded.valid_labels,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args, encoded).to(device)

    label_counts = torch.bincount(encoded.train_labels)
    class_weights = label_counts.sum() / label_counts.clamp(min=1)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    print("device:", device)
    print("Train size:", len(train_dataset))
    print("Valid size:", len(valid_dataset))
    print("Classes:", encoded.label_encoder.classes_.tolist())
    print("Model variant:", args.model_variant)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for categorical_inputs, numeric_inputs, labels in train_loader:
            categorical_inputs = categorical_inputs.to(device)
            numeric_inputs = numeric_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(categorical_inputs, numeric_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            running_correct += (predictions == labels).sum().item()
            running_examples += labels.size(0)

        train_loss = running_loss / max(running_examples, 1)
        train_acc = running_correct / max(running_examples, 1)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, device, criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": val_loss,
                "valid_accuracy": val_acc,
            }
        )

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"valid_loss={val_loss:.4f} "
            f"valid_acc={val_acc:.4f}"
        )

        improved = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) <= 1e-9 and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_loss, valid_acc, valid_labels, valid_predictions = evaluate(
        model, valid_loader, device, criterion
    )
    class_names = encoded.label_encoder.classes_.tolist()
    report = classification_report(
        valid_labels,
        valid_predictions,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    print(f"Best validation accuracy: {valid_acc:.4f}")
    print(
        classification_report(
            valid_labels,
            valid_predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    artifact = {
        "target": args.target,
        "model_variant": args.model_variant,
        "classes": class_names,
        "categorical_columns": encoded.categorical_columns,
        "numeric_columns": encoded.numeric_columns,
        "categorical_vocab_sizes": encoded.categorical_vocab_sizes,
        "max_rows": args.max_rows,
        "top_apps": args.top_apps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "attention_dim": args.attention_dim,
        "attention_heads": args.attention_heads,
        "attention_layers": args.attention_layers,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "best_validation_accuracy": valid_acc,
        "best_validation_loss": valid_loss,
        "history": history,
        "report": report,
        "numeric_stats": encoded.numeric_stats,
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": artifact,
        "label_classes": class_names,
        "category_maps": encoded.category_maps,
        "numeric_stats": encoded.numeric_stats,
    }
    torch.save(checkpoint, args.model_path)

    with args.report_path.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=True, indent=2)

    print(f"Saved model to: {args.model_path}")
    print(f"Saved report to: {args.report_path}")
