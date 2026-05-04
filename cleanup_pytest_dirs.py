from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_NAMES = {
    ".pytest_cache",
}
TARGET_PREFIXES = (
    "pytest-",
    "pytest_",
    "pytesttemp",
    "pytest-temp",
)
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
}


def should_delete(path: Path) -> bool:
    name = path.name
    if name in TARGET_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in TARGET_PREFIXES)


def iter_pytest_dirs(root: Path) -> list[Path]:
    matches: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue

        for child in children:
            if not child.is_dir():
                continue
            if child.name in SKIP_DIR_NAMES:
                continue
            if should_delete(child):
                matches.append(child)
                continue
            stack.append(child)

    return sorted(matches, key=lambda item: (len(item.parts), str(item)))


def remove_pytest_dirs(root: Path, dry_run: bool) -> int:
    targets = iter_pytest_dirs(root)
    if not targets:
        print("[cleanup] no pytest temporary directories found")
        return 0

    deleted = 0
    for target in targets:
        print(f"[cleanup] {'would remove' if dry_run else 'removing'}: {target}")
        if dry_run:
            continue
        try:
            shutil.rmtree(target)
            deleted += 1
        except OSError as exc:
            print(f"[cleanup] failed: {target} -> {exc}")

    if not dry_run:
        print(f"[cleanup] removed {deleted} directories")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove pytest temporary directories from the current project."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root to scan. Defaults to the repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matching directories without deleting them.",
    )
    return parser


def run_cleanup(root: Path, dry_run: bool) -> int:
    root = root.resolve()
    print(f"[cleanup] scanning: {root}")
    return remove_pytest_dirs(root, dry_run=dry_run)


def main() -> int:
    args = build_parser().parse_args()
    return run_cleanup(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
