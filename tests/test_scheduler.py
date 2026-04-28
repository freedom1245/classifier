from cdc_priority.scheduler.env import SchedulerEnv
from pathlib import Path
import shutil
import uuid

import pandas as pd

from cdc_priority.scheduler.evaluate import (
    export_policy_comparison,
    export_policy_comparison_figure,
    simulate_policy,
)
from cdc_priority.scheduler.agent import PPOAgent
from cdc_priority.scheduler.event import CDCEvent
from cdc_priority.scheduler.reward import compute_reward
from cdc_priority.scheduler.training import _is_better_validation_candidate
from cdc_priority.scheduler.policies import (
    aging_light_policy,
    aging_policy,
    high_guarded_aging_policy,
    low_rescue_policy,
    strict_priority_policy,
)
from cdc_priority.scheduler.queue_manager import QueueManager


def _make_scheduler_temp_dir() -> Path:
    path = Path("outputs/scheduler") / f"pytest-temp-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_scheduler_env_reset() -> None:
    env = SchedulerEnv()
    state = env.reset()
    assert state.queue_length == 0
    assert state.priority_counts == {"high": 0, "medium": 0, "low": 0}
    assert state.low_queue_fraction == 0.0
    assert state.starvation_pressure == 0.0


def test_scheduler_state_masks_aggressive_low_priority_actions_when_high_is_waiting() -> None:
    state = SchedulerEnv().reset()
    state.queue_length = 10
    state.priority_counts = {"high": 4, "medium": 1, "low": 5}
    state.max_low_priority_wait_steps = 24
    state.low_queue_fraction = 0.5
    state.starvation_pressure = 1.2

    allowed = state.allowed_actions(starvation_threshold=20)

    assert 5 not in allowed
    assert 3 not in allowed
    assert 4 in allowed


def test_strict_priority_policy_prefers_high_priority() -> None:
    queue_manager = QueueManager()
    queue_manager.push(CDCEvent("e1", "low", arrival_step=0, sync_cost=1.0))
    queue_manager.push(CDCEvent("e2", "high", arrival_step=1, sync_cost=1.0))

    event = strict_priority_policy(queue_manager)

    assert event is not None
    assert event.event_id == "e2"


def test_aging_policy_boosts_waiting_low_priority_event() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("low_old", "low", arrival_step=0, sync_cost=1.0, wait_steps=10)
    )
    queue_manager.push(
        CDCEvent("medium_new", "medium", arrival_step=1, sync_cost=1.0, wait_steps=1)
    )

    event = aging_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "low_old"


def test_aging_policy_still_prefers_high_priority_when_wait_is_similar() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("low_waiting", "low", arrival_step=0, sync_cost=1.0, wait_steps=6)
    )
    queue_manager.push(
        CDCEvent("high_waiting", "high", arrival_step=1, sync_cost=1.0, wait_steps=6)
    )

    event = aging_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "high_waiting"


def test_aging_policy_rescues_severely_starved_low_priority_event() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("high_new", "high", arrival_step=0, sync_cost=1.0, wait_steps=2)
    )
    queue_manager.push(
        CDCEvent("low_starved", "low", arrival_step=1, sync_cost=1.0, wait_steps=12)
    )

    event = aging_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "low_starved"


def test_aging_light_is_more_conservative_than_strong_aging() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("high_new", "high", arrival_step=0, sync_cost=1.0, wait_steps=2)
    )
    queue_manager.push(
        CDCEvent("low_waiting", "low", arrival_step=1, sync_cost=1.0, wait_steps=6)
    )

    event = aging_light_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "high_new"


def test_high_guarded_aging_keeps_high_when_low_not_far_enough_ahead() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("high_waiting", "high", arrival_step=0, sync_cost=1.0, wait_steps=8)
    )
    queue_manager.push(
        CDCEvent("low_waiting", "low", arrival_step=1, sync_cost=1.0, wait_steps=12)
    )

    event = high_guarded_aging_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "high_waiting"


def test_low_rescue_policy_forces_low_once_threshold_is_crossed() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("high_waiting", "high", arrival_step=0, sync_cost=1.0, wait_steps=2)
    )
    queue_manager.push(
        CDCEvent("low_starved", "low", arrival_step=1, sync_cost=1.0, wait_steps=6)
    )

    event = low_rescue_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "low_starved"


def test_simulate_policy_returns_metrics() -> None:
    events = [
        CDCEvent("e1", "low", arrival_step=0, sync_cost=1.0, service_steps=1),
        CDCEvent("e2", "high", arrival_step=1, sync_cost=1.0, service_steps=1),
        CDCEvent("e3", "medium", arrival_step=2, sync_cost=1.0, service_steps=1),
    ]

    metrics = simulate_policy(events, policy_name="strict_priority", starvation_threshold=5)

    assert metrics.completed_events == 3
    assert metrics.throughput > 0
    assert metrics.average_delay_steps >= 0


def test_export_policy_comparison_writes_csv() -> None:
    temp_dir = _make_scheduler_temp_dir()
    try:
        data_path = temp_dir / "events.csv"
        output_path = temp_dir / "policy_comparison.csv"
        frame = pd.DataFrame(
            {
                "event_id": ["e1", "e2", "e3"],
                "timestamp": [
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:02",
                    "2024-01-01 00:00:03",
                ],
                "priority_label": ["low", "high", "medium"],
                "estimated_sync_cost": [1.0, 1.0, 1.0],
                "deadline": [3.0, 2.0, 4.0],
            }
        )
        frame.to_csv(data_path, index=False)

        comparison = export_policy_comparison(
            data_path,
            output_path,
            starvation_threshold=5,
        )

        assert output_path.exists()
        assert set(comparison["policy"]) == {"fifo", "strict_priority", "aging"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_export_policy_comparison_uses_cache_when_input_unchanged() -> None:
    temp_dir = _make_scheduler_temp_dir()
    try:
        data_path = temp_dir / "events.csv"
        output_path = temp_dir / "policy_comparison.csv"
        frame = pd.DataFrame(
            {
                "event_id": ["e1", "e2", "e3"],
                "timestamp": [
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:02",
                    "2024-01-01 00:00:03",
                ],
                "priority_label": ["low", "high", "medium"],
                "estimated_sync_cost": [1.0, 1.0, 1.0],
                "deadline": [3.0, 2.0, 4.0],
            }
        )
        frame.to_csv(data_path, index=False)

        first = export_policy_comparison(data_path, output_path, starvation_threshold=5)
        first_mtime = output_path.stat().st_mtime_ns
        second = export_policy_comparison(data_path, output_path, starvation_threshold=5)
        second_mtime = output_path.stat().st_mtime_ns

        assert set(first["policy"]) == {"fifo", "strict_priority", "aging"}
        assert second.equals(first)
        assert second_mtime == first_mtime
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_export_policy_comparison_figure_writes_image() -> None:
    temp_dir = _make_scheduler_temp_dir()
    try:
        csv_path = temp_dir / "policy_comparison.csv"
        figure_path = temp_dir / "policy_comparison.png"
        pd.DataFrame(
            {
                "policy": ["fifo", "strict_priority", "aging"],
                "throughput": [0.1, 0.1, 0.1],
                "average_delay_steps": [100, 80, 90],
                "high_priority_average_delay_steps": [100, 10, 20],
                "max_low_priority_wait_steps": [200, 300, 250],
                "fairness_index": [1.0, 0.5, 0.6],
                "completed_events": [3, 3, 3],
            }
        ).to_csv(csv_path, index=False)

        output = export_policy_comparison_figure(csv_path, figure_path)

        assert output.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_compute_reward_rewards_reducing_low_wait() -> None:
    before = SchedulerEnv().reset()
    before.max_low_priority_wait_steps = 20
    before.queue_length = 4
    before.priority_counts["low"] = 2
    before.low_queue_fraction = 0.5
    before.starvation_pressure = 1.0
    after = SchedulerEnv().reset()
    after.max_low_priority_wait_steps = 5
    after.queue_length = 3
    after.priority_counts["low"] = 1
    after.low_queue_fraction = 1 / 3
    after.starvation_pressure = 0.25

    reward = compute_reward(
        previous_state=before,
        state=after,
        action=2,
        processed_priority="low",
        processed_delay_steps=2,
        deadline_missed=False,
        starvation_threshold=20,
        reward_weights={
            "high_priority_throughput": 1.0,
            "average_delay_penalty": 0.05,
            "starvation_penalty": 3.0,
            "deadline_miss_penalty": 2.0,
        },
    )

    assert reward > 0


def test_compute_reward_prefers_aging_when_starvation_is_active() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 6
    before.priority_counts = {"high": 2, "medium": 1, "low": 3}
    before.max_low_priority_wait_steps = 30
    before.low_queue_fraction = 0.5
    before.starvation_pressure = 1.5

    after = SchedulerEnv().reset()
    after.queue_length = 5
    after.priority_counts = {"high": 2, "medium": 1, "low": 2}
    after.max_low_priority_wait_steps = 18
    after.low_queue_fraction = 0.4
    after.starvation_pressure = 0.9

    common_kwargs = {
        "previous_state": before,
        "state": after,
        "processed_priority": "low",
        "processed_delay_steps": 4,
        "deadline_missed": False,
        "starvation_threshold": 20,
        "reward_weights": {
            "high_priority_throughput": 1.0,
            "high_priority_delay_penalty": 0.08,
            "average_delay_penalty": 0.05,
            "starvation_penalty": 3.0,
            "deadline_miss_penalty": 2.0,
            "aging_starvation_bonus": 3.0,
            "aging_balancing_bonus": 1.0,
            "aging_high_priority_penalty": 3.0,
            "high_backlog_service_bonus": 2.0,
            "guarded_high_service_bonus": 1.5,
            "guarded_mixed_bonus": 1.5,
            "strict_high_pressure_bonus": 1.0,
            "defer_high_penalty": 2.0,
            "light_aging_high_penalty": 2.0,
            "low_rescue_high_penalty": 3.0,
            "low_rescue_emergency_bonus": 1.5,
            "low_rescue_overuse_penalty": 2.0,
            "strict_under_starvation_penalty": 2.0,
            "fifo_mixed_pressure_penalty": 2.5,
            "low_priority_service_bonus": 1.0,
            "low_queue_reduction_bonus": 0.75,
        },
    }

    aging_reward = compute_reward(action=2, **common_kwargs)
    strict_reward = compute_reward(action=1, **common_kwargs)

    assert aging_reward > strict_reward


def test_compute_reward_penalizes_fifo_when_high_and_starved_low_coexist() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 8
    before.priority_counts = {"high": 3, "medium": 1, "low": 4}
    before.max_low_priority_wait_steps = 28
    before.low_queue_fraction = 0.5
    before.starvation_pressure = 1.4

    after = SchedulerEnv().reset()
    after.queue_length = 7
    after.priority_counts = {"high": 3, "medium": 1, "low": 3}
    after.max_low_priority_wait_steps = 20
    after.low_queue_fraction = 3 / 7
    after.starvation_pressure = 1.0

    reward_kwargs = {
        "previous_state": before,
        "state": after,
        "processed_priority": "low",
        "processed_delay_steps": 3,
        "deadline_missed": False,
        "starvation_threshold": 20,
        "reward_weights": {
            "high_priority_throughput": 1.0,
            "high_priority_delay_penalty": 0.08,
            "average_delay_penalty": 0.05,
            "starvation_penalty": 3.0,
            "deadline_miss_penalty": 2.0,
            "aging_starvation_bonus": 3.0,
            "aging_balancing_bonus": 1.0,
            "aging_high_priority_penalty": 3.0,
            "high_backlog_service_bonus": 2.0,
            "guarded_high_service_bonus": 1.5,
            "guarded_mixed_bonus": 1.5,
            "strict_high_pressure_bonus": 1.0,
            "defer_high_penalty": 2.0,
            "light_aging_high_penalty": 2.0,
            "low_rescue_high_penalty": 3.0,
            "low_rescue_emergency_bonus": 1.5,
            "low_rescue_overuse_penalty": 2.0,
            "strict_under_starvation_penalty": 2.0,
            "fifo_mixed_pressure_penalty": 2.5,
            "low_priority_service_bonus": 1.0,
            "low_queue_reduction_bonus": 0.75,
        },
    }

    fifo_reward = compute_reward(action=0, **reward_kwargs)
    aging_reward = compute_reward(action=2, **reward_kwargs)

    assert aging_reward > fifo_reward


def test_compute_reward_penalizes_aging_when_high_queue_is_waiting() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 10
    before.priority_counts = {"high": 5, "medium": 1, "low": 4}
    before.max_low_priority_wait_steps = 30
    before.low_queue_fraction = 0.4
    before.starvation_pressure = 1.5

    after = SchedulerEnv().reset()
    after.queue_length = 9
    after.priority_counts = {"high": 5, "medium": 1, "low": 3}
    after.max_low_priority_wait_steps = 20
    after.low_queue_fraction = 1 / 3
    after.starvation_pressure = 1.0

    reward_kwargs = {
        "previous_state": before,
        "state": after,
        "processed_delay_steps": 3,
        "deadline_missed": False,
        "starvation_threshold": 20,
        "reward_weights": {
            "high_priority_throughput": 1.0,
            "high_priority_delay_penalty": 0.08,
            "average_delay_penalty": 0.05,
            "starvation_penalty": 3.0,
            "deadline_miss_penalty": 2.0,
            "aging_starvation_bonus": 3.0,
            "aging_balancing_bonus": 1.0,
            "aging_high_priority_penalty": 3.0,
            "high_backlog_service_bonus": 2.0,
            "guarded_high_service_bonus": 1.5,
            "guarded_mixed_bonus": 1.5,
            "strict_high_pressure_bonus": 1.0,
            "defer_high_penalty": 2.0,
            "light_aging_high_penalty": 2.0,
            "low_rescue_high_penalty": 3.0,
            "low_rescue_emergency_bonus": 1.5,
            "low_rescue_overuse_penalty": 2.0,
            "strict_under_starvation_penalty": 2.0,
            "fifo_mixed_pressure_penalty": 2.5,
            "low_priority_service_bonus": 1.0,
            "low_queue_reduction_bonus": 0.75,
        },
    }

    aging_low_reward = compute_reward(
        action=2,
        processed_priority="low",
        **reward_kwargs,
    )
    aging_high_reward = compute_reward(
        action=2,
        processed_priority="high",
        **reward_kwargs,
    )

    assert aging_high_reward > aging_low_reward


def test_compute_reward_prefers_guarded_high_service_under_mixed_pressure() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 10
    before.priority_counts = {"high": 4, "medium": 1, "low": 5}
    before.max_low_priority_wait_steps = 28
    before.low_queue_fraction = 0.5
    before.starvation_pressure = 1.4

    after = SchedulerEnv().reset()
    after.queue_length = 9
    after.priority_counts = {"high": 3, "medium": 1, "low": 5}
    after.max_low_priority_wait_steps = 18
    after.low_queue_fraction = 5 / 9
    after.starvation_pressure = 0.9

    reward_kwargs = {
        "previous_state": before,
        "state": after,
        "processed_delay_steps": 3,
        "deadline_missed": False,
        "starvation_threshold": 20,
        "reward_weights": {
            "high_priority_throughput": 1.0,
            "high_priority_delay_penalty": 0.08,
            "average_delay_penalty": 0.05,
            "starvation_penalty": 3.0,
            "deadline_miss_penalty": 2.0,
            "aging_starvation_bonus": 3.0,
            "aging_balancing_bonus": 1.0,
            "aging_high_priority_penalty": 3.0,
            "high_backlog_service_bonus": 2.0,
            "guarded_high_service_bonus": 1.5,
            "guarded_mixed_bonus": 1.5,
            "strict_high_pressure_bonus": 1.0,
            "defer_high_penalty": 2.0,
            "light_aging_high_penalty": 2.0,
            "low_rescue_high_penalty": 3.0,
            "low_rescue_emergency_bonus": 1.5,
            "low_rescue_overuse_penalty": 2.0,
            "strict_under_starvation_penalty": 2.0,
            "fifo_mixed_pressure_penalty": 2.5,
            "low_priority_service_bonus": 1.0,
            "low_queue_reduction_bonus": 0.75,
        },
    }

    guarded_high_reward = compute_reward(
        action=4,
        processed_priority="high",
        **reward_kwargs,
    )
    low_rescue_reward = compute_reward(
        action=5,
        processed_priority="low",
        **reward_kwargs,
    )

    assert guarded_high_reward > low_rescue_reward


def test_compute_reward_penalizes_low_rescue_when_not_in_emergency_mode() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 10
    before.priority_counts = {"high": 4, "medium": 1, "low": 5}
    before.max_low_priority_wait_steps = 24
    before.low_queue_fraction = 0.5
    before.starvation_pressure = 1.2

    after = SchedulerEnv().reset()
    after.queue_length = 9
    after.priority_counts = {"high": 4, "medium": 1, "low": 4}
    after.max_low_priority_wait_steps = 20
    after.low_queue_fraction = 4 / 9
    after.starvation_pressure = 1.0

    reward_kwargs = {
        "previous_state": before,
        "state": after,
        "processed_delay_steps": 3,
        "deadline_missed": False,
        "starvation_threshold": 20,
        "reward_weights": {
            "high_priority_throughput": 1.0,
            "high_priority_delay_penalty": 0.08,
            "average_delay_penalty": 0.05,
            "starvation_penalty": 3.0,
            "deadline_miss_penalty": 2.0,
            "aging_starvation_bonus": 3.0,
            "aging_balancing_bonus": 1.0,
            "aging_high_priority_penalty": 3.0,
            "high_backlog_service_bonus": 2.0,
            "guarded_high_service_bonus": 1.5,
            "guarded_mixed_bonus": 1.5,
            "strict_high_pressure_bonus": 1.0,
            "defer_high_penalty": 2.0,
            "light_aging_high_penalty": 2.0,
            "low_rescue_high_penalty": 3.0,
            "low_rescue_emergency_bonus": 1.5,
            "low_rescue_overuse_penalty": 2.0,
            "strict_under_starvation_penalty": 2.0,
            "fifo_mixed_pressure_penalty": 2.5,
            "low_priority_service_bonus": 1.0,
            "low_queue_reduction_bonus": 0.75,
        },
    }

    low_rescue_reward = compute_reward(
        action=5,
        processed_priority="low",
        **reward_kwargs,
    )
    guarded_high_reward = compute_reward(
        action=4,
        processed_priority="high",
        **reward_kwargs,
    )

    assert guarded_high_reward > low_rescue_reward


def test_ppo_agent_can_sample_and_optimize() -> None:
    agent = PPOAgent(
        action_count=3,
        state_dim=8,
        hidden_dim=32,
        update_epochs=1,
        mini_batch_size=2,
        device="cpu",
    )

    for index in range(4):
        state = [float(index), 1.0, 2.0, 3.0, 4.0, 5.0, 0.2, 0.4]
        action, log_prob, value = agent.select_action(state, deterministic=False)
        assert 0 <= action < 3
        agent.store_transition(
            state=state,
            action=action,
            log_prob=log_prob,
            reward=1.0,
            done=index == 3,
            value=value,
        )

    loss = agent.optimize(last_value=0.0)

    assert isinstance(loss, float)


def test_ppo_agent_respects_allowed_action_mask() -> None:
    agent = PPOAgent(
        action_count=6,
        state_dim=8,
        hidden_dim=32,
        update_epochs=1,
        mini_batch_size=2,
        device="cpu",
    )

    action, _, _ = agent.select_action(
        [10.0, 20.0, 4.0, 1.0, 5.0, 24.0, 0.5, 1.2],
        deterministic=False,
        allowed_actions=[0, 1, 2, 4],
    )

    assert action in {0, 1, 2, 4}


def test_validation_candidate_selection_prefers_lower_high_delay_when_rewards_are_close() -> None:
    best = {
        "validation_reward": -1000.0,
        "validation_average_delay_steps": 100.0,
        "validation_high_priority_average_delay_steps": 80.0,
        "validation_fairness_index": 0.9,
    }
    candidate = {
        "validation_reward": -900.0,
        "validation_average_delay_steps": 105.0,
        "validation_high_priority_average_delay_steps": 40.0,
        "validation_fairness_index": 0.88,
    }

    assert _is_better_validation_candidate(
        candidate,
        best,
        reward_tolerance=200.0,
        high_delay_tolerance=10.0,
        fairness_tolerance=1e-4,
    )
