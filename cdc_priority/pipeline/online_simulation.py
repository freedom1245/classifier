from __future__ import annotations

import json
from pathlib import Path

from ..classifier.training import run_classifier_training
from ..data.dataset_builder import (
    build_and_export_dataset_from_config,
    build_and_export_scheduler_dataset_from_config,
)
from ..scheduler.training import run_scheduler_training
from ..settings import default_settings, load_yaml_config
from ..utils import generate_run_name, resolve_run_output_dir


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_pipeline(
    classifier_config: Path,
    scheduler_config: Path,
    run_name: str | None = None,
) -> Path:
    settings = default_settings()
    classifier_config = _resolve_project_path(settings.project_root, classifier_config)
    scheduler_config = _resolve_project_path(settings.project_root, scheduler_config)

    classifier_bundle = load_yaml_config(classifier_config)
    scheduler_bundle = load_yaml_config(scheduler_config)
    classifier_values = classifier_bundle.values
    scheduler_values = scheduler_bundle.values

    classifier_dataset_config = _resolve_project_path(
        settings.project_root,
        classifier_values["dataset_config"],
    )
    scheduler_dataset_config = _resolve_project_path(
        settings.project_root,
        scheduler_values.get("dataset_config", classifier_values["dataset_config"]),
    )

    shared_run_name = run_name or str(classifier_values.get("run_name") or "").strip() or str(
        scheduler_values.get("run_name") or ""
    ).strip() or generate_run_name("pipeline")

    classifier_dataset_output_dir = settings.project_root / "data" / "processed"
    scheduler_dataset_output_dir = _resolve_project_path(
        settings.project_root,
        scheduler_values["scheduler_dataset_dir"],
    )
    classifier_output_base_dir = _resolve_project_path(
        settings.project_root,
        classifier_values["output_dir"],
    )
    scheduler_output_base_dir = _resolve_project_path(
        settings.project_root,
        scheduler_values["output_dir"],
    )
    classifier_output_dir, _ = resolve_run_output_dir(
        classifier_output_base_dir,
        shared_run_name,
        prefix="classifier",
    )
    scheduler_output_dir, _ = resolve_run_output_dir(
        scheduler_output_base_dir,
        shared_run_name,
        prefix=str(scheduler_values.get("algorithm", "scheduler")).lower(),
    )
    pipeline_output_dir, _ = resolve_run_output_dir(
        settings.outputs_dir / "pipeline",
        shared_run_name,
        prefix="pipeline",
    )

    classifier_random_state = int(classifier_values.get("random_state", 42))
    scheduler_random_state = int(scheduler_values.get("random_state", 42))

    print("[pipeline] step 1/5: export classifier dataset")
    classifier_prepared = build_and_export_dataset_from_config(
        classifier_dataset_config,
        classifier_dataset_output_dir,
        random_state=classifier_random_state,
    )

    print("[pipeline] step 2/5: export scheduler dataset")
    scheduler_prepared = build_and_export_scheduler_dataset_from_config(
        scheduler_dataset_config,
        scheduler_dataset_output_dir,
        timestamp_column="timestamp",
    )

    print("[pipeline] step 3/5: train classifier")
    run_classifier_training(classifier_config, run_name=shared_run_name)

    print("[pipeline] step 4/5: train scheduler")
    run_scheduler_training(scheduler_config, run_name=shared_run_name)

    print("[pipeline] step 5/5: aggregate reports")
    classifier_report = _read_json(classifier_output_dir / "classifier_report.json")
    scheduler_report = _read_json(scheduler_output_dir / "scheduler_report.json")

    pipeline_report = {
        "classifier_config": str(classifier_config),
        "scheduler_config": str(scheduler_config),
        "classifier_dataset_config": str(classifier_dataset_config),
        "scheduler_dataset_config": str(scheduler_dataset_config),
        "paths": {
            "classifier_dataset_output_dir": str(classifier_dataset_output_dir),
            "scheduler_dataset_output_dir": str(scheduler_dataset_output_dir),
            "classifier_output_dir": str(classifier_output_dir),
            "scheduler_output_dir": str(scheduler_output_dir),
            "pipeline_output_dir": str(pipeline_output_dir),
        },
        "run_name": shared_run_name,
        "classifier_dataset_report": classifier_prepared.report,
        "scheduler_dataset_report": scheduler_prepared.report,
        "classifier_summary": {
            "model_variant": classifier_report.get("model_variant"),
            "validation": classifier_report.get("validation"),
            "test": classifier_report.get("test"),
        },
        "scheduler_summary": {
            "algorithm": scheduler_report.get("algorithm"),
            "best_validation_reward": scheduler_report.get("best_validation_reward"),
            "best_validation_summary": scheduler_report.get("best_validation_summary"),
            "policy_comparison": scheduler_report.get("policy_comparison"),
        },
    }
    if classifier_dataset_config != scheduler_dataset_config:
        pipeline_report["warnings"] = [
            "Classifier and scheduler are using different dataset config files."
        ]

    pipeline_report_path = pipeline_output_dir / "pipeline_report.json"
    with pipeline_report_path.open("w", encoding="utf-8") as file:
        json.dump(pipeline_report, file, ensure_ascii=True, indent=2)

    print(f"[pipeline] classifier dataset rows: {classifier_prepared.report['row_count']}")
    print(f"[pipeline] scheduler dataset rows: {scheduler_prepared.report['row_count']}")
    print(f"[pipeline] classifier report: {classifier_output_dir / 'classifier_report.json'}")
    print(f"[pipeline] scheduler report: {scheduler_output_dir / 'scheduler_report.json'}")
    print(f"[pipeline] pipeline report: {pipeline_report_path}")
    return pipeline_report_path
