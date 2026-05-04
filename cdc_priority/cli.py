import argparse
from pathlib import Path
from typing import Sequence

from .classifier.training import run_classifier_ablations, run_classifier_training
from .data.dataset_builder import (
    build_and_export_dataset_from_config,
    build_and_export_scheduler_dataset_from_config,
)
from .pipeline.online_simulation import run_pipeline
from .scheduler.evaluate import (
    export_policy_comparison,
    export_policy_comparison_figure,
)
from .scheduler.training import run_scheduler_training
from .settings import default_settings


def build_parser() -> argparse.ArgumentParser:
    settings = default_settings()
    parser = argparse.ArgumentParser(
        description="CDC priority classification and RL scheduling toolkit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classifier_parser = subparsers.add_parser("classifier")
    classifier_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "classifier.yaml",
        help="Classifier YAML config path.",
    )
    classifier_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output subdirectory name for this classifier run.",
    )

    classifier_ablation_parser = subparsers.add_parser("classifier-ablation")
    classifier_ablation_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "classifier.yaml",
        help="Classifier YAML config path.",
    )
    classifier_ablation_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output subdirectory name for this ablation run.",
    )

    scheduler_parser = subparsers.add_parser("scheduler")
    scheduler_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "scheduler.yaml",
        help="Scheduler YAML config path.",
    )
    scheduler_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output subdirectory name for this scheduler run.",
    )

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "dataset.yaml",
        help="Dataset YAML config path.",
    )
    dataset_parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.project_root / "data" / "processed",
        help="Where to export train/valid/test splits and dataset report.",
    )
    dataset_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for dataset splitting.",
    )

    scheduler_dataset_parser = subparsers.add_parser("scheduler-dataset")
    scheduler_dataset_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "dataset.yaml",
        help="Dataset YAML config path.",
    )
    scheduler_dataset_parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.project_root / "data" / "scheduler_processed",
        help="Where to export time-ordered scheduler train/valid/test splits and report.",
    )
    scheduler_dataset_parser.add_argument(
        "--timestamp-column",
        type=str,
        default="timestamp",
        help="Timestamp column used for time-ordered scheduler splitting.",
    )

    scheduler_visualize_parser = subparsers.add_parser("scheduler-visualize")
    scheduler_visualize_parser.add_argument(
        "--data",
        type=Path,
        default=settings.project_root / "data" / "scheduler_processed" / "test.csv",
        help="Scheduler evaluation dataset path.",
    )
    scheduler_visualize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.project_root / "outputs" / "scheduler",
        help="Where to export scheduler comparison tables and figures.",
    )
    scheduler_visualize_parser.add_argument(
        "--starvation-threshold",
        type=int,
        default=20,
        help="Starvation threshold passed to the aging policy.",
    )

    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument(
        "--classifier-config",
        type=Path,
        default=settings.configs_dir / "classifier.yaml",
        help="Classifier YAML config path.",
    )
    pipeline_parser.add_argument(
        "--scheduler-config",
        type=Path,
        default=settings.configs_dir / "scheduler.yaml",
        help="Scheduler YAML config path.",
    )
    pipeline_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional shared output subdirectory name for classifier, scheduler, and pipeline reports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "classifier":
        run_classifier_training(args.config, run_name=args.run_name)
        return

    if args.command == "classifier-ablation":
        run_classifier_ablations(args.config, run_name=args.run_name)
        return

    if args.command == "scheduler":
        run_scheduler_training(args.config, run_name=args.run_name)
        return

    if args.command == "dataset":
        prepared = build_and_export_dataset_from_config(
            args.config,
            args.output_dir,
            random_state=args.random_state,
        )
        print(f"[dataset] exported train split to: {args.output_dir / 'train.csv'}")
        print(f"[dataset] exported valid split to: {args.output_dir / 'valid.csv'}")
        print(f"[dataset] exported test split to: {args.output_dir / 'test.csv'}")
        print(f"[dataset] exported report to: {args.output_dir / 'dataset_report.json'}")
        print(f"[dataset] row count: {prepared.report['row_count']}")
        return

    if args.command == "scheduler-dataset":
        prepared = build_and_export_scheduler_dataset_from_config(
            args.config,
            args.output_dir,
            timestamp_column=args.timestamp_column,
        )
        print(
            f"[scheduler-dataset] exported train split to: {args.output_dir / 'train.csv'}"
        )
        print(
            f"[scheduler-dataset] exported valid split to: {args.output_dir / 'valid.csv'}"
        )
        print(
            f"[scheduler-dataset] exported test split to: {args.output_dir / 'test.csv'}"
        )
        print(
            f"[scheduler-dataset] exported report to: {args.output_dir / 'dataset_report.json'}"
        )
        print(f"[scheduler-dataset] split strategy: {prepared.report['split_strategy']}")
        return

    if args.command == "scheduler-visualize":
        comparison_path = args.output_dir / "policy_comparison.csv"
        comparison = export_policy_comparison(
            data_path=args.data,
            output_path=comparison_path,
            starvation_threshold=args.starvation_threshold,
        )
        comparison_figure_path = export_policy_comparison_figure(
            comparison_csv_path=comparison_path,
            output_path=args.output_dir / "policy_comparison.png",
        )
        print(f"[scheduler-visualize] exported comparison to: {comparison_path}")
        print(
            f"[scheduler-visualize] exported comparison figure to: {comparison_figure_path}"
        )
        print(
            f"[scheduler-visualize] compared policies: {', '.join(comparison['policy'].tolist())}"
        )
        return

    run_pipeline(args.classifier_config, args.scheduler_config, run_name=args.run_name)
