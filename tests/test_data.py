import shutil
from pathlib import Path
import uuid

import pandas as pd
import pytest

from cdc_priority.classifier.features import encode_dataset
from cdc_priority.data.dataset_builder import (
    PreparedDataset,
    build_report_with_metadata,
    build_scheduler_dataset,
    export_prepared_dataset,
)
from cdc_priority.data.labeler import attach_priority_label
from cdc_priority.data.loader import load_events
from cdc_priority.data.schema import DatasetSchema
from cdc_priority.data.splitter import DatasetSplit


def _make_workspace_temp_dir() -> Path:
    path = Path("data/processed") / f"pytest-temp-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_schema_feature_columns() -> None:
    schema = DatasetSchema(
        target="priority_label",
        categorical_columns=["event_type"],
        numeric_columns=["record_size"],
    )
    assert schema.feature_columns == ["event_type", "record_size"]


def test_attach_priority_label_builds_expected_classes() -> None:
    frame = pd.DataFrame(
        {
            "event_type": ["payment", "update", "log"],
            "business_domain": ["order", "profile", "archive"],
            "business_value": [10, 5, 0],
            "queue_wait_time": [8, 2, 0],
            "dependency_count": [5, 1, 0],
            "estimated_sync_cost": [1, 4, 10],
            "retry_count": [1, 0, 0],
        }
    )

    labeled = attach_priority_label(frame)

    assert "priority_score" in labeled.columns
    assert labeled.loc[0, "priority_label"] == "high"
    assert labeled.loc[2, "priority_label"] == "low"


def test_attach_priority_label_honors_configurable_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "event_type": ["payment", "update", "log"],
            "business_domain": ["order", "profile", "archive"],
            "business_value": [10, 6, 0],
            "queue_wait_time": [8, 4, 0],
            "dependency_count": [5, 2, 0],
            "estimated_sync_cost": [1, 3, 10],
            "retry_count": [1, 1, 0],
        }
    )

    labeled = attach_priority_label(
        frame,
        labeling_config={
            "priority_score": {
                "numeric_weights": {
                    "business_value": 0.60,
                    "queue_wait_time": 0.40,
                },
                "categorical_weights": {},
                "hot_values": {},
                "invert_numeric": [],
                "thresholds": {
                    "medium": 0.30,
                    "high": 0.80,
                }
            }
        },
    )

    assert labeled.loc[0, "priority_label"] == "high"
    assert labeled.loc[1, "priority_label"] == "medium"


def test_attach_priority_label_supports_component_scoring() -> None:
    frame = pd.DataFrame(
        {
            "business_value": [100, 20, 5],
            "business_domain": ["electronics", "archive", "archive"],
            "user_level": ["vip", "normal", "normal"],
            "queue_wait_time": [9, 2, 0],
            "deadline_gap": [0.5, 4.0, 8.0],
            "retry_count": [1, 0, 0],
            "is_peak_hour": [1, 0, 0],
            "dependency_count": [5, 1, 0],
            "consistency_risk": [0.9, 0.2, 0.1],
            "event_type": ["purchase", "view", "view"],
            "estimated_sync_cost": [1.0, 4.0, 9.0],
            "source_load": [0.2, 0.5, 0.9],
            "db_load": [0.3, 0.5, 0.9],
        }
    )

    labeled = attach_priority_label(
        frame,
        labeling_config={
            "priority_score": {
                "components": {
                    "business_importance": {
                        "weight": 0.4,
                        "numeric_weights": {"business_value": 0.75},
                        "categorical_weights": {"business_domain": 0.15, "user_level": 0.10},
                        "invert_numeric": [],
                        "hot_values": {
                            "business_domain": ["electronics"],
                            "user_level": ["vip"],
                        },
                    },
                    "timeliness": {
                        "weight": 0.3,
                        "numeric_weights": {
                            "queue_wait_time": 0.45,
                            "deadline_gap": 0.35,
                            "retry_count": 0.10,
                            "is_peak_hour": 0.10,
                        },
                        "invert_numeric": ["deadline_gap"],
                        "categorical_weights": {},
                        "hot_values": {},
                    },
                    "dependency_impact": {
                        "weight": 0.2,
                        "numeric_weights": {
                            "dependency_count": 0.6,
                            "consistency_risk": 0.4,
                        },
                        "categorical_weights": {"event_type": 1.0},
                        "invert_numeric": [],
                        "hot_values": {"event_type": ["purchase"]},
                    },
                    "execution_feasibility": {
                        "weight": 0.1,
                        "numeric_weights": {
                            "estimated_sync_cost": 0.7,
                            "source_load": 0.15,
                            "db_load": 0.15,
                        },
                        "invert_numeric": ["estimated_sync_cost", "source_load", "db_load"],
                        "categorical_weights": {},
                        "hot_values": {},
                    },
                },
                "thresholds": {
                    "medium": 0.34,
                    "high": 0.50,
                },
            }
        },
    )

    assert labeled.loc[0, "priority_label"] == "high"
    assert labeled.loc[2, "priority_label"] == "low"


def test_encode_dataset_produces_train_valid_test_tensors() -> None:
    frame = pd.DataFrame(
        {
            "event_type": ["payment", "update", "log", "payment", "update", "log"],
            "table_name": ["orders", "users", "logs", "orders", "users", "logs"],
            "business_domain": ["order", "profile", "archive", "order", "profile", "archive"],
            "record_size": [100, 50, 10, 120, 40, 8],
            "estimated_sync_cost": [1, 4, 10, 1, 3, 8],
            "dependency_count": [5, 1, 0, 4, 1, 0],
            "queue_wait_time": [8, 2, 0, 7, 1, 0],
            "priority_label": ["high", "medium", "low", "high", "medium", "low"],
        }
    )
    schema = DatasetSchema(
        target="priority_label",
        categorical_columns=["event_type", "table_name", "business_domain"],
        numeric_columns=[
            "record_size",
            "estimated_sync_cost",
            "dependency_count",
            "queue_wait_time",
        ],
    )
    prepared = PreparedDataset(
        schema=schema,
        full_frame=frame,
        split=DatasetSplit(
            train=frame.iloc[:3].copy(),
            valid=frame.iloc[3:5].copy(),
            test=frame.iloc[5:].copy(),
        ),
        report={},
    )

    encoded = encode_dataset(prepared)

    assert encoded.train.categorical.shape == (3, 3)
    assert encoded.valid.numeric.shape == (2, 4)
    assert encoded.test.labels.shape == (1,)


def test_dataset_report_contains_labeling_metadata() -> None:
    frame = pd.DataFrame({"priority_label": ["high", "medium", "low"]})
    split = DatasetSplit(
        train=frame.iloc[:1].copy(),
        valid=frame.iloc[1:2].copy(),
        test=frame.iloc[2:].copy(),
    )

    report = build_report_with_metadata(
        frame,
        split,
        target="priority_label",
        labeling_config={
            "priority_score": {
                "thresholds": {
                    "medium": 0.30,
                    "high": 0.80,
                }
            }
        },
    )

    assert "labeling" in report
    assert report["labeling"]["priority_score"]["thresholds"]["high"] == 0.80


def test_load_events_supports_csv_and_json() -> None:
    temp_dir = _make_workspace_temp_dir()
    try:
        frame = pd.DataFrame(
            {
                "event_type": ["payment", "update"],
                "priority_label": ["high", "medium"],
            }
        )
        csv_path = temp_dir / "events.csv"
        json_path = temp_dir / "events.json"
        frame.to_csv(csv_path, index=False)
        frame.to_json(json_path, orient="records")

        csv_loaded = load_events(csv_path)
        json_loaded = load_events(json_path)

        assert list(csv_loaded.columns) == list(frame.columns)
        assert list(json_loaded["priority_label"]) == ["high", "medium"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_events_supports_parquet_when_engine_available() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    assert pyarrow is not None

    temp_dir = _make_workspace_temp_dir()
    try:
        frame = pd.DataFrame(
            {
                "event_type": ["payment"],
                "priority_label": ["high"],
            }
        )
        parquet_path = temp_dir / "events.parquet"
        frame.to_parquet(parquet_path, index=False)

        loaded = load_events(parquet_path)

        assert loaded.loc[0, "priority_label"] == "high"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_export_prepared_dataset_writes_expected_files() -> None:
    temp_dir = _make_workspace_temp_dir()
    try:
        frame = pd.DataFrame(
            {
                "event_type": ["payment", "update", "log"],
                "priority_label": ["high", "medium", "low"],
            }
        )
        prepared = PreparedDataset(
            schema=DatasetSchema(
                target="priority_label",
                categorical_columns=["event_type"],
                numeric_columns=[],
            ),
            full_frame=frame,
            split=DatasetSplit(
                train=frame.iloc[:1].copy(),
                valid=frame.iloc[1:2].copy(),
                test=frame.iloc[2:].copy(),
            ),
            report={"row_count": 3},
        )

        export_prepared_dataset(prepared, temp_dir)

        assert (temp_dir / "train.csv").exists()
        assert (temp_dir / "valid.csv").exists()
        assert (temp_dir / "test.csv").exists()
        assert (temp_dir / "dataset_report.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_scheduler_dataset_uses_time_ordered_split() -> None:
    temp_dir = _make_workspace_temp_dir()
    try:
        source_path = temp_dir / "events.csv"
        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-01 00:00:03",
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:02",
                    "2024-01-01 00:00:04",
                ],
                "event_type": ["view", "cart", "purchase", "view"],
                "table_name": ["a", "b", "c", "d"],
                "business_domain": ["electronics", "electronics", "computers", "appliances"],
                "record_size": [1, 2, 3, 4],
                "estimated_sync_cost": [1.0, 1.0, 1.0, 1.0],
                "dependency_count": [1, 1, 1, 1],
                "queue_wait_time": [1.0, 1.0, 1.0, 1.0],
                "priority_label": ["low", "medium", "high", "low"],
            }
        )
        frame.to_csv(source_path, index=False)
        schema = DatasetSchema(
            target="priority_label",
            categorical_columns=["event_type", "table_name", "business_domain"],
            numeric_columns=[
                "record_size",
                "estimated_sync_cost",
                "dependency_count",
                "queue_wait_time",
            ],
        )

        prepared = build_scheduler_dataset(
            data_path=source_path,
            schema=schema,
            train_ratio=0.5,
            valid_ratio=0.25,
            test_ratio=0.25,
            timestamp_column="timestamp",
        )

        assert prepared.report["split_strategy"] == "time_ordered"
        assert str(prepared.split.train.iloc[0]["timestamp"]) == "2024-01-01 00:00:01"
        assert str(prepared.split.test.iloc[-1]["timestamp"]) == "2024-01-01 00:00:04"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
