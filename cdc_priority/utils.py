from datetime import datetime
from pathlib import Path
import re


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_run_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip(".-")
    return sanitized or "run"


def generate_run_name(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{sanitize_run_name(prefix)}-{timestamp}"


def resolve_run_output_dir(
    base_dir: Path,
    run_name: str | None = None,
    prefix: str = "run",
) -> tuple[Path, str]:
    ensure_directory(base_dir)
    resolved_run_name = sanitize_run_name(run_name) if run_name else generate_run_name(prefix)
    output_dir = ensure_directory(base_dir / resolved_run_name)
    return output_dir, resolved_run_name
