# CDC Priority RL

Graduation-design project for CDC event priority classification and
reinforcement-learning-based synchronization scheduling.

## Standard Workflow

1. Build the classification dataset:
   - `python build_dataset.py`
2. Build the scheduler dataset:
   - `python build_scheduler_dataset.py`
3. Train the priority classifier:
   - `python train_classifier.py`
4. Train the scheduler:
   - `python train_scheduler.py`
5. Run the end-to-end pipeline:
   - `python run_pipeline.py`

By default, each classifier, scheduler, and pipeline run writes to a timestamped
subdirectory under its configured `output_dir`. Use `--run-name <name>` when you
want multiple commands to share the same artifact folder.

## Entry Points

- `python build_dataset.py`: build the classification dataset with random train/valid/test splits
- `python build_scheduler_dataset.py`: build the scheduler dataset with time-ordered splits
- `python train_classifier.py`: train the tabular priority classifier
- `python train_classifier.py --run-name experiment-a`: train the classifier into `outputs/classifier/experiment-a/`
- `python train_scheduler.py`: train the DQN-based scheduler
- `python train_scheduler.py --run-name experiment-a`: train the scheduler into its own `experiment-a` subdirectory
- `python train_scheduler.py --config configs/scheduler_ppo.yaml`: train the PPO scheduler
- `python train_scheduler.py --config configs/scheduler_double_dqn.yaml`: train the Double DQN scheduler
- `python -m cdc_priority.cli classifier-ablation --config configs/classifier.yaml --run-name experiment-a`: run classifier ablation experiments
- `python cleanup_pytest_dirs.py --dry-run`: preview pytest temporary directories before removing them
- `python cleanup_pytest_dirs.py`: remove pytest temporary directories
- `python -m cdc_priority.cli scheduler-visualize`: export scheduler comparison tables and figures
- `python run_pipeline.py --run-name experiment-a`: export datasets, train classifier, train scheduler, and write a pipeline summary report

## Tests

- `python -m pytest tests/test_data.py -q -p no:cacheprovider`
- `python -m pytest tests/test_classifier.py -q -p no:cacheprovider`
- `python -m pytest tests/test_scheduler.py -q -p no:cacheprovider`
- `python -m pytest tests/test_pipeline.py -q -p no:cacheprovider`

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
- `configs/scheduler_ppo.yaml` and `configs/scheduler_double_dqn.yaml` are alternative scheduler experiment configs
- Training and pipeline outputs are grouped under run-specific subdirectories to avoid overwriting earlier experiments
- Generated datasets, model checkpoints, reports, and figures are treated as local artifacts and are ignored by Git
