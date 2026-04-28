# CDC Priority RL

Graduation-design project for CDC event priority classification and
reinforcement-learning-based synchronization scheduling.

## Entry Points

- `python build_dataset.py`: build the classification dataset with random train/valid/test splits
- `python build_scheduler_dataset.py`: build the scheduler dataset with time-ordered splits
- `python train_classifier.py`: train the tabular priority classifier
- `python train_scheduler.py`: train the DQN-based scheduler
- `python -m cdc_priority.cli scheduler-visualize --skip-timeline`: export scheduler comparison results faster during iteration
- `python run_pipeline.py`: reserved for end-to-end pipeline execution

## Structure

- `configs/`: YAML configuration files
- `cdc_priority/data/`: dataset loading, preprocessing, labeling, splitting
- `cdc_priority/classifier/`: tabular classification models and evaluation
- `cdc_priority/scheduler/`: scheduling policies, RL environment, fairness logic
- `cdc_priority/pipeline/`: end-to-end orchestration
- `outputs/`: generated models, metrics, and figures

## Notes

The project now uses the `cdc_priority` package as the primary implementation
for dataset preparation, classification, and scheduling experiments.

- Classification uses `configs/dataset.yaml` and `configs/classifier.yaml`
- Scheduling uses `configs/scheduler.yaml` and `data/scheduler_processed/`
- Generated datasets, model checkpoints, reports, and figures are treated as local artifacts and are ignored by Git
