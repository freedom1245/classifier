from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .event import CDCEvent
from .policies import aging_policy, fifo_policy, strict_priority_policy
from .queue_manager import QueueManager


@dataclass
class SchedulerMetrics:
    throughput: float
    average_delay_steps: float
    max_low_priority_wait_steps: float
    completed_events: int
    high_priority_average_delay_steps: float
    fairness_index: float


def load_scheduler_events(data_path: Path) -> list[CDCEvent]:
    frame = pd.read_csv(data_path)
    if "timestamp" in frame.columns:
        # 调度实验必须尊重事件到达顺序，因此这里强制按时间排序。
        frame = frame.sort_values("timestamp").reset_index(drop=True)

    events: list[CDCEvent] = []
    for index, row in frame.iterrows():
        deadline_value = row["deadline"] if "deadline" in frame.columns else None
        deadline_step = None if pd.isna(deadline_value) else int(math.ceil(float(deadline_value)))
        # 当前仿真里 estimated_sync_cost 被近似映射为离散 service_steps。
        service_steps = max(1, int(math.ceil(float(row.get("estimated_sync_cost", 1.0)))))
        events.append(
            CDCEvent(
                event_id=str(row.get("event_id", f"event_{index}")),
                priority=str(row.get("priority_label", "low")),
                arrival_step=index,
                sync_cost=float(row.get("estimated_sync_cost", 1.0)),
                deadline_step=deadline_step,
                service_steps=service_steps,
            )
        )
    return events


def _select_event(
    queue_manager: QueueManager,
    policy_name: str,
    starvation_threshold: int,
) -> CDCEvent | None:
    if policy_name == "fifo":
        return fifo_policy(queue_manager)
    if policy_name == "strict_priority":
        return strict_priority_policy(queue_manager)
    if policy_name == "aging":
        return aging_policy(queue_manager, starvation_threshold=starvation_threshold)
    raise ValueError(f"Unsupported policy: {policy_name}")


def _jain_fairness(values: list[float]) -> float:
    filtered = [value for value in values if value >= 0]
    if not filtered:
        return 0.0
    numerator = sum(filtered) ** 2
    denominator = len(filtered) * sum(value * value for value in filtered)
    return 0.0 if denominator <= 1e-12 else numerator / denominator


def simulate_policy(
    events: list[CDCEvent],
    policy_name: str,
    starvation_threshold: int = 5,
) -> SchedulerMetrics:
    # 这里是“固定策略”的离线重放仿真，用来和 RL 策略做同口径比较。
    queue_manager = QueueManager()
    current_step = 0
    next_index = 0
    completed = 0
    delay_totals: list[int] = []
    high_delay_totals: list[int] = []
    per_priority_delay: dict[str, list[int]] = {"high": [], "medium": [], "low": []}

    while next_index < len(events) or len(queue_manager) > 0:
        while next_index < len(events) and events[next_index].arrival_step <= current_step:
            event = events[next_index]
            queue_manager.push(
                CDCEvent(
                    event_id=event.event_id,
                    priority=event.priority,
                    arrival_step=event.arrival_step,
                    sync_cost=event.sync_cost,
                    deadline_step=event.deadline_step,
                    wait_steps=0,
                    service_steps=event.service_steps,
                )
            )
            next_index += 1

        if len(queue_manager) == 0:
            current_step = events[next_index].arrival_step
            continue

        event = _select_event(
            queue_manager,
            policy_name=policy_name,
            starvation_threshold=starvation_threshold,
        )
        if event is None:
            current_step += 1
            continue

        delay = current_step - event.arrival_step
        delay_totals.append(delay)
        per_priority_delay[event.priority].append(delay)
        if event.priority == "high":
            high_delay_totals.append(delay)
        completed += 1

        queue_manager.increment_wait_steps(event.service_steps)
        current_step += event.service_steps

    average_delay_steps = sum(delay_totals) / max(len(delay_totals), 1)
    high_average_delay_steps = sum(high_delay_totals) / max(len(high_delay_totals), 1)
    max_low_priority_wait_steps = max(per_priority_delay["low"], default=0)
    fairness_index = _jain_fairness(
        [
            sum(per_priority_delay["high"]) / max(len(per_priority_delay["high"]), 1),
            sum(per_priority_delay["medium"]) / max(len(per_priority_delay["medium"]), 1),
            sum(per_priority_delay["low"]) / max(len(per_priority_delay["low"]), 1),
        ]
    )
    throughput = completed / max(current_step, 1)
    return SchedulerMetrics(
        throughput=throughput,
        average_delay_steps=average_delay_steps,
        max_low_priority_wait_steps=max_low_priority_wait_steps,
        completed_events=completed,
        high_priority_average_delay_steps=high_average_delay_steps,
        fairness_index=fairness_index,
    )


def compare_policies(
    events: list[CDCEvent],
    starvation_threshold: int = 5,
) -> pd.DataFrame:
    rows = []
    for policy_name in ("fifo", "strict_priority", "aging"):
        metrics = simulate_policy(
            events,
            policy_name=policy_name,
            starvation_threshold=starvation_threshold,
        )
        rows.append(
            {
                "policy": policy_name,
                "throughput": metrics.throughput,
                "average_delay_steps": metrics.average_delay_steps,
                "high_priority_average_delay_steps": metrics.high_priority_average_delay_steps,
                "max_low_priority_wait_steps": metrics.max_low_priority_wait_steps,
                "fairness_index": metrics.fairness_index,
                "completed_events": metrics.completed_events,
            }
        )
    return pd.DataFrame(rows)


def _cache_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def _is_comparison_cache_valid(
    data_path: Path,
    output_path: Path,
    starvation_threshold: int,
) -> bool:
    metadata_path = _cache_metadata_path(output_path)
    if not output_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("data_path") == str(data_path.resolve())
        and metadata.get("starvation_threshold") == starvation_threshold
        and metadata.get("data_mtime_ns") == data_path.stat().st_mtime_ns
    )


def _write_comparison_cache_metadata(
    data_path: Path,
    output_path: Path,
    starvation_threshold: int,
) -> None:
    metadata_path = _cache_metadata_path(output_path)
    metadata = {
        "data_path": str(data_path.resolve()),
        "starvation_threshold": starvation_threshold,
        "data_mtime_ns": data_path.stat().st_mtime_ns,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def export_policy_comparison(
    data_path: Path,
    output_path: Path,
    starvation_threshold: int = 5,
) -> pd.DataFrame:
    if _is_comparison_cache_valid(data_path, output_path, starvation_threshold):
        return pd.read_csv(output_path)

    events = load_scheduler_events(data_path)
    comparison = compare_policies(events, starvation_threshold=starvation_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    _write_comparison_cache_metadata(data_path, output_path, starvation_threshold)
    return comparison


def export_policy_comparison_figure(
    comparison_csv_path: Path,
    output_path: Path,
) -> Path:
    comparison = pd.read_csv(comparison_csv_path)
    metrics = [
        ("throughput", "Throughput"),
        ("average_delay_steps", "Average Delay (steps)"),
        ("high_priority_average_delay_steps", "High-Priority Delay (steps)"),
        ("fairness_index", "Fairness Index"),
    ]
    colors = {
        "fifo": "#3B82F6",
        "strict_priority": "#EF4444",
        "aging": "#10B981",
        "dqn": "#F59E0B",
        "ppo": "#8B5CF6",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    policies = comparison["policy"].tolist()

    for axis, (metric_key, title) in zip(axes, metrics):
        values = comparison[metric_key].tolist()
        axis.bar(
            policies,
            values,
            color=[colors.get(policy, "#6B7280") for policy in policies],
        )
        axis.set_title(title)
        axis.set_xlabel("Policy")
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Scheduler Policy Comparison", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def append_policy_result(
    comparison_csv_path: Path,
    row: dict[str, object],
) -> pd.DataFrame:
    comparison = pd.read_csv(comparison_csv_path)
    comparison = comparison[comparison["policy"] != row["policy"]].copy()
    comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
    comparison.to_csv(comparison_csv_path, index=False)
    return comparison
