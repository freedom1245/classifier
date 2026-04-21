from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch


def build_priority_label(frame: pd.DataFrame) -> pd.Series:
    priority = pd.Series("low", index=frame.index, dtype="object")

    medium_mask = (
        frame["service_class"].eq("THROUGHPUT_ORIENTED")
        | frame["io_zone"].eq("WARM")
        | frame["cache_hit"].eq(1)
    )
    high_mask = (
        frame["service_class"].eq("LATENCY_SENSITIVE")
        | (frame["io_zone"].eq("WARM") & frame["cache_hit"].eq(1))
    )

    priority.loc[medium_mask] = "medium"
    priority.loc[high_mask] = "high"
    return priority


def load_data(data_path: Path, max_rows: int) -> pd.DataFrame:
    nrows = None if max_rows <= 0 else max_rows
    frame = pd.read_csv(data_path, nrows=nrows)

    numeric_columns = [
        "start_time",
        "c_time",
        "simulated_disk_start_time",
        "file_offset",
        "from_flash_cache",
        "cache_hit",
        "request_io_size_bytes",
        "disk_io_size_bytes",
        "response_io_size_bytes",
        "disk_time",
        "simulated_latency",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["file_age_seconds"] = (frame["start_time"] - frame["c_time"]).clip(lower=0)
    frame["queue_delay_seconds"] = (
        frame["simulated_disk_start_time"] - frame["start_time"]
    ).clip(lower=0)

    seconds_in_day = 24 * 60 * 60
    time_of_day = frame["start_time"].fillna(0) % seconds_in_day
    frame["start_hour"] = (time_of_day // 3600).astype("int64")
    frame["start_minute"] = ((time_of_day % 3600) // 60).astype("int64")

    frame["application"] = frame["application"].fillna("UNKNOWN")
    frame["io_zone"] = frame["io_zone"].fillna("UNKNOWN")
    frame["redundancy_type"] = frame["redundancy_type"].fillna("UNKNOWN")
    frame["op_type"] = frame["op_type"].fillna("UNKNOWN")
    frame["service_class"] = frame["service_class"].fillna("OTHER")
    return frame


def collapse_rare_applications(frame: pd.DataFrame, top_apps: int) -> pd.DataFrame:
    frame = frame.copy()
    top_values = set(frame["application"].value_counts().head(top_apps).index)
    frame["application_grouped"] = frame["application"].where(
        frame["application"].isin(top_values), "__OTHER__"
    )
    return frame


def select_target(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "priority_label":
        return build_priority_label(frame)
    return frame[target].astype(str)


def encode_categorical_column(
    train_series: pd.Series, valid_series: pd.Series
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    vocab_values = sorted(train_series.astype(str).unique().tolist())
    vocab = {"__UNK__": 0}
    for index, value in enumerate(vocab_values, start=1):
        vocab[value] = index

    train_encoded = train_series.astype(str).map(lambda value: vocab.get(value, 0))
    valid_encoded = valid_series.astype(str).map(lambda value: vocab.get(value, 0))

    return (
        vocab,
        torch.tensor(train_encoded.to_numpy(), dtype=torch.long),
        torch.tensor(valid_encoded.to_numpy(), dtype=torch.long),
    )


def zscore_standardize(
    train_frame: pd.DataFrame, valid_frame: pd.DataFrame, numeric_columns: list[str]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    train_numeric = train_frame[numeric_columns].copy().fillna(0)
    valid_numeric = valid_frame[numeric_columns].copy().fillna(0)

    means = train_numeric.mean()
    stds = train_numeric.std().replace(0, 1.0).fillna(1.0)

    train_scaled = ((train_numeric - means) / stds).astype("float32")
    valid_scaled = ((valid_numeric - means) / stds).astype("float32")

    stats = {
        column: {"mean": float(means[column]), "std": float(stds[column])}
        for column in numeric_columns
    }
    return (
        torch.tensor(train_scaled.to_numpy(), dtype=torch.float32),
        torch.tensor(valid_scaled.to_numpy(), dtype=torch.float32),
        stats,
    )


@dataclass
class EncodedData:
    train_categorical: torch.Tensor
    valid_categorical: torch.Tensor
    train_numeric: torch.Tensor
    valid_numeric: torch.Tensor
    train_labels: torch.Tensor
    valid_labels: torch.Tensor
    categorical_columns: list[str]
    numeric_columns: list[str]
    categorical_vocab_sizes: list[int]
    category_maps: dict[str, dict[str, int]]
    label_encoder: LabelEncoder
    numeric_stats: dict[str, dict[str, float]]


def prepare_encoded_data(
    frame: pd.DataFrame, target: str, test_size: float, random_state: int
) -> EncodedData:
    y = select_target(frame, target)

    feature_columns = [
        "application_grouped",
        "io_zone",
        "redundancy_type",
        "op_type",
        "from_flash_cache",
        "cache_hit",
        "file_offset",
        "file_age_seconds",
        "request_io_size_bytes",
        "disk_io_size_bytes",
        "response_io_size_bytes",
        "disk_time",
        "simulated_latency",
        "queue_delay_seconds",
        "start_hour",
        "start_minute",
    ]
    categorical_columns = [
        "application_grouped",
        "io_zone",
        "redundancy_type",
        "op_type",
    ]
    if target == "io_zone":
        feature_columns.remove("io_zone")
        categorical_columns.remove("io_zone")

    X = frame[feature_columns].copy()
    X["from_flash_cache"] = pd.to_numeric(
        X["from_flash_cache"], errors="coerce"
    ).fillna(0)
    X["cache_hit"] = pd.to_numeric(X["cache_hit"], errors="coerce").fillna(0)

    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=random_state,
    )

    category_maps: dict[str, dict[str, int]] = {}
    train_cat_tensors = []
    valid_cat_tensors = []
    categorical_vocab_sizes = []

    for column in categorical_columns:
        vocab, train_encoded, valid_encoded = encode_categorical_column(
            X_train[column], X_valid[column]
        )
        category_maps[column] = vocab
        categorical_vocab_sizes.append(len(vocab))
        train_cat_tensors.append(train_encoded.unsqueeze(1))
        valid_cat_tensors.append(valid_encoded.unsqueeze(1))

    if train_cat_tensors:
        train_categorical = torch.cat(train_cat_tensors, dim=1)
        valid_categorical = torch.cat(valid_cat_tensors, dim=1)
    else:
        train_categorical = torch.zeros((len(X_train), 0), dtype=torch.long)
        valid_categorical = torch.zeros((len(X_valid), 0), dtype=torch.long)

    train_numeric, valid_numeric, numeric_stats = zscore_standardize(
        X_train, X_valid, numeric_columns
    )

    return EncodedData(
        train_categorical=train_categorical,
        valid_categorical=valid_categorical,
        train_numeric=train_numeric,
        valid_numeric=valid_numeric,
        train_labels=torch.tensor(y_train, dtype=torch.long),
        valid_labels=torch.tensor(y_valid, dtype=torch.long),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        categorical_vocab_sizes=categorical_vocab_sizes,
        category_maps=category_maps,
        label_encoder=label_encoder,
        numeric_stats=numeric_stats,
    )
