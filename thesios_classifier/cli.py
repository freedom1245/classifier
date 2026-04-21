try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc

from .config import parse_args
from .data import collapse_rare_applications, load_data, prepare_encoded_data
from .training import set_seed, train_and_save


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    print(f"Loading data from: {args.data_path}")
    frame = load_data(args.data_path, args.max_rows)
    frame = collapse_rare_applications(frame, args.top_apps)
    encoded = prepare_encoded_data(
        frame=frame,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_and_save(args, encoded)
