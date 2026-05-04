import json
import shutil
from pathlib import Path
import uuid

import pandas as pd
import yaml

from cdc_priority.pipeline import online_simulation as pipeline_module
from cdc_priority.settings import AppSettings, load_yaml_config


def _make_pipeline_temp_dir() -> Path:
    path = (Path("outputs/pipeline") / f"pytest-temp-{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_run_pipeline_exports_datasets_and_aggregates_reports(monkeypatch) -> None:
    temp_root = _make_pipeline_temp_dir()
    try:
        project_root = temp_root
        configs_dir = project_root / "configs"
        raw_dir = project_root / "raw"
        configs_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:02",
                    "2024-01-01 00:00:03",
                    "2024-01-01 00:00:04",
                    "2024-01-01 00:00:05",
                    "2024-01-01 00:00:06",
                    "2024-01-01 00:00:07",
                    "2024-01-01 00:00:08",
                    "2024-01-01 00:00:09",
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:00:11",
                    "2024-01-01 00:00:12",
                ],
                "event_type": [
                    "view", "cart", "purchase", "view", "cart", "purchase",
                    "view", "cart", "purchase", "view", "cart", "purchase",
                ],
                "table_name": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"],
                "business_domain": [
                    "electronics", "electronics", "computers", "electronics", "appliances", "computers",
                    "electronics", "appliances", "computers", "electronics", "electronics", "computers",
                ],
                "source_service": [
                    "brand1", "brand2", "brand3", "brand1", "brand2", "brand3",
                    "brand1", "brand2", "brand3", "brand1", "brand2", "brand3",
                ],
                "user_level": [
                    "normal", "vip", "normal", "normal", "vip", "internal",
                    "normal", "vip", "internal", "normal", "vip", "normal",
                ],
                "record_size": [10, 12, 15, 10, 12, 15, 11, 13, 16, 12, 14, 17],
                "changed_columns_count": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                "estimated_sync_cost": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
                "business_value": [50, 80, 120, 55, 85, 125, 52, 82, 122, 58, 88, 128],
                "consistency_risk": [0.2, 0.4, 0.8, 0.2, 0.4, 0.8, 0.25, 0.45, 0.85, 0.3, 0.5, 0.9],
                "dependency_count": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                "queue_wait_time": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                "deadline": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                "retry_count": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                "source_load": [0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4],
                "db_load": [0.2, 0.25, 0.35, 0.2, 0.25, 0.35, 0.2, 0.25, 0.35, 0.2, 0.25, 0.35],
                "is_peak_hour": [0] * 12,
                "is_hot_data": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                "priority_label": [
                    "low", "medium", "high", "low", "medium", "high",
                    "low", "medium", "high", "low", "medium", "high",
                ],
            }
        )
        raw_data_path = raw_dir / "events.csv"
        frame.to_csv(raw_data_path, index=False)

        dataset_config_path = configs_dir / "dataset.yaml"
        classifier_config_path = configs_dir / "classifier.yaml"
        scheduler_config_path = configs_dir / "scheduler.yaml"

        dataset_config = {
            "data_path": str(raw_data_path),
            "target": "priority_label",
            "categorical_columns": [
                "event_type",
                "table_name",
                "business_domain",
                "source_service",
                "user_level",
            ],
            "numeric_columns": [
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
            "split": {"train": 0.5, "valid": 0.25, "test": 0.25},
        }
        classifier_output_dir = project_root / "outputs" / "classifier"
        scheduler_output_dir = project_root / "outputs" / "scheduler"
        scheduler_dataset_dir = project_root / "data" / "scheduler_processed"

        classifier_config = {
            "source": "configured_dataset",
            "dataset_config": str(dataset_config_path),
            "model_variant": "embedding_mlp",
            "batch_size": 8,
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "hidden_dim": 32,
            "dropout": 0.1,
            "embedding_dim": 8,
            "numeric_projection_dim": 16,
            "attention_dim": 16,
            "attention_heads": 2,
            "attention_layers": 1,
            "patience": 1,
            "num_workers": 0,
            "random_state": 42,
            "output_dir": str(classifier_output_dir),
        }
        scheduler_config = {
            "dataset_config": str(dataset_config_path),
            "scheduler_dataset_dir": str(scheduler_dataset_dir),
            "classifier_output_dir": str(classifier_output_dir),
            "algorithm": "dqn",
            "episode_count": 1,
            "max_steps_per_episode": 10,
            "train_event_window": 4,
            "valid_event_window": 2,
            "random_state": 42,
            "queue_capacity": 100,
            "starvation_threshold": 5,
            "defer_low_priority": True,
            "low_release_batch_size": 2,
            "low_release_load_threshold": 0.35,
            "low_release_high_queue_threshold": 2,
            "low_force_release_wait_steps": 10,
            "off_peak_start_hour": 0,
            "off_peak_end_hour": 6,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "batch_size": 4,
            "replay_capacity": 32,
            "target_update_interval": 5,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.99,
            "reward_weights": {
                "throughput_weight": 1.0,
                "delay_weight": 0.1,
                "fairness_weight": 0.5,
                "deferred_low_weight": 0.05,
                "off_peak_release_weight": 0.1,
                "deadline_weight": 2.0,
            },
            "output_dir": str(scheduler_output_dir),
        }

        dataset_config_path.write_text(yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8")
        classifier_config_path.write_text(
            yaml.safe_dump(classifier_config, sort_keys=False),
            encoding="utf-8",
        )
        scheduler_config_path.write_text(
            yaml.safe_dump(scheduler_config, sort_keys=False),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            pipeline_module,
            "default_settings",
            lambda: AppSettings(
                project_root=project_root,
                configs_dir=configs_dir,
                outputs_dir=project_root / "outputs",
            ),
        )

        def fake_run_classifier_training(config_path: Path, run_name: str | None = None) -> Path:
            bundle = load_yaml_config(config_path)
            output_dir = Path(bundle.values["output_dir"]) / (run_name or "run")
            output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "model_variant": bundle.values["model_variant"],
                "validation": {"accuracy": 0.9, "macro_f1": 0.9},
                "test": {"accuracy": 0.91, "macro_f1": 0.91},
            }
            (output_dir / "classifier_report.json").write_text(
                json.dumps(report, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            return output_dir

        def fake_run_scheduler_training(config_path: Path, run_name: str | None = None) -> Path:
            bundle = load_yaml_config(config_path)
            output_dir = Path(bundle.values["output_dir"]) / (run_name or "run")
            output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "algorithm": bundle.values["algorithm"],
                "best_validation_reward": 1.23,
                "best_validation_summary": {
                    "validation_reward": 1.23,
                    "validation_high_priority_average_delay_steps": 2.0,
                    "validation_average_delay_steps": 3.0,
                    "validation_fairness_index": 0.8,
                },
                "policy_comparison": [
                    {"policy": "fifo", "average_delay_steps": 4.0},
                    {"policy": "dqn", "average_delay_steps": 3.0},
                ],
            }
            (output_dir / "scheduler_report.json").write_text(
                json.dumps(report, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            return output_dir

        monkeypatch.setattr(pipeline_module, "run_classifier_training", fake_run_classifier_training)
        monkeypatch.setattr(pipeline_module, "run_scheduler_training", fake_run_scheduler_training)

        pipeline_module.run_pipeline(
            classifier_config_path,
            scheduler_config_path,
            run_name="pytest_pipeline",
        )

        assert (project_root / "data" / "processed" / "train.csv").exists()
        assert (project_root / "data" / "scheduler_processed" / "test.csv").exists()
        assert (classifier_output_dir / "pytest_pipeline" / "classifier_report.json").exists()
        assert (scheduler_output_dir / "pytest_pipeline" / "scheduler_report.json").exists()

        pipeline_report_path = (
            project_root / "outputs" / "pipeline" / "pytest_pipeline" / "pipeline_report.json"
        )
        assert pipeline_report_path.exists()

        pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
        assert pipeline_report["classifier_summary"]["model_variant"] == "embedding_mlp"
        assert pipeline_report["scheduler_summary"]["algorithm"] == "dqn"
        assert pipeline_report["classifier_dataset_report"]["row_count"] == 12
        assert pipeline_report["scheduler_dataset_report"]["split_strategy"] == "time_ordered"
        assert pipeline_report["run_name"] == "pytest_pipeline"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
