import argparse
from pathlib import Path


DEFAULT_DATASET = "thesios_cluster1_16TB_20240115_data-00000-of-00100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch multi-class classifier on the Thesios trace."
    )
    parser.add_argument(
        "--model-variant",
        choices=["v1", "v2"],
        default="v2",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(DEFAULT_DATASET),
        help="Path to the Thesios CSV shard.",
    )
    parser.add_argument(
        "--target",
        choices=["service_class", "io_zone", "priority_label"],
        default="priority_label",
        help="Target column to train on.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Read at most this many rows. Use 0 to read the full shard.",
    )
    parser.add_argument(
        "--top-apps",
        type=int,
        default=64,
        help="Keep the top-N applications as explicit categories.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden width of the tabular network.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--attention-dim",
        type=int,
        default=128,
        help="Hidden size used by the self-attention feature encoder.",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=8,
        help="Number of attention heads in each self-attention block.",
    )
    parser.add_argument(
        "--attention-layers",
        type=int,
        default=3,
        help="Number of stacked self-attention blocks.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("thesios_torch_model.pt"),
        help="Where to save the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("thesios_torch_report.json"),
        help="Where to save the training report.",
    )
    return parser.parse_args()
