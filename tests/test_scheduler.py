from cdc_priority.scheduler.env import SchedulerEnv
from pathlib import Path
import shutil
import uuid

import pandas as pd

from cdc_priority.scheduler.evaluate import (
    export_policy_comparison,
    export_policy_comparison_figure,
    load_scheduler_events,
    simulate_policy,
)
from cdc_priority.scheduler.agent import DoubleDQNAgent, PPOAgent
from cdc_priority.scheduler.event import CDCEvent
from cdc_priority.scheduler.reward import compute_reward
from cdc_priority.scheduler.training import (
    _is_better_validation_candidate,
    run_scheduler_training,
)
from cdc_priority.settings import AppSettings
import json
import yaml
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
    assert state.max_high_wait == 0
    assert state.max_medium_wait == 0
    assert state.deadline_missed_count == 0
    assert state.wait_steps_std == 0.0
    assert state.deferred_low_count == 0
    assert len(state.to_vector()) == 19


def test_scheduler_env_defers_low_priority_during_peak_hours() -> None:
    env = SchedulerEnv(
        events=[
            CDCEvent(
                "low_peak",
                "low",
                arrival_step=0,
                sync_cost=1.0,
                arrival_hour=12,
            )
        ],
        queue_capacity=100,
        defer_low_priority=True,
        off_peak_start_hour=0,
        off_peak_end_hour=6,
    )

    state = env.reset()

    assert state.priority_counts["low"] == 0
    assert state.deferred_low_count == 1
    assert state.is_off_peak == 0.0


def test_scheduler_env_releases_deferred_low_during_off_peak_window() -> None:
    env = SchedulerEnv(
        events=[
            CDCEvent(
                "low_off_peak",
                "low",
                arrival_step=0,
                sync_cost=1.0,
                arrival_hour=2,
            )
        ],
        queue_capacity=100,
        defer_low_priority=True,
        off_peak_start_hour=0,
        off_peak_end_hour=6,
    )

    state = env.reset()

    assert state.priority_counts["low"] == 1
    assert state.deferred_low_count == 0
    assert state.is_off_peak == 1.0


def test_scheduler_env_not_done_while_deferred_low_events_remain() -> None:
    env = SchedulerEnv(
        events=[
            CDCEvent(
                "low_peak",
                "low",
                arrival_step=0,
                sync_cost=1.0,
                arrival_hour=12,
            )
        ],
        queue_capacity=100,
        defer_low_priority=True,
        off_peak_start_hour=0,
        off_peak_end_hour=6,
        low_force_release_wait_steps=1000,
    )

    env.reset()
    _, _, done, _ = env.step(0)

    assert done is False


def test_scheduler_state_masks_aggressive_low_priority_actions_when_high_is_waiting() -> None:
    state = SchedulerEnv().reset()
    state.queue_length = 10
    state.priority_counts = {"high": 4, "medium": 1, "low": 5}
    state.max_low_priority_wait_steps = 24
    state.low_queue_fraction = 0.5
    state.starvation_pressure = 1.2
    state.max_high_wait = 10
    state.max_medium_wait = 5
    state.deadline_missed_count = 0
    state.wait_steps_std = 5.0
    state.remaining_fraction = 0.8
    state.avg_sync_cost = 1.5

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


def test_load_scheduler_events_uses_real_timestamp_gaps_for_arrival_steps() -> None:
    temp_dir = _make_scheduler_temp_dir()
    try:
        data_path = temp_dir / "events.csv"
        frame = pd.DataFrame(
            {
                "event_id": ["e1", "e2", "e3"],
                "timestamp": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:10",
                ],
                "priority_label": ["low", "medium", "high"],
                "estimated_sync_cost": [1.0, 1.0, 1.0],
            }
        )
        frame.to_csv(data_path, index=False)

        events = load_scheduler_events(data_path)

        assert [event.arrival_step for event in events] == [0, 1, 10]
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


def test_compute_reward_throughput_increases_with_priority() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 3
    before.max_low_priority_wait_steps = 10
    before.max_high_wait = 10
    before.max_medium_wait = 10
    before.wait_steps_std = 2.0
    after = SchedulerEnv().reset()
    after.queue_length = 2
    after.max_low_priority_wait_steps = 8
    after.max_high_wait = 8
    after.max_medium_wait = 8
    after.wait_steps_std = 3.0

    weights = {
        "throughput_weight": 1.0,
        "delay_weight": 0.1,
        "fairness_weight": 0.5,
        "deadline_weight": 2.0,
    }
    high_reward = compute_reward(
        previous_state=before, state=after,
        action=0, processed_priority="high", processed_delay_steps=3,
        deadline_missed=False, reward_weights=weights,
    )
    medium_reward = compute_reward(
        previous_state=before, state=after,
        action=0, processed_priority="medium", processed_delay_steps=3,
        deadline_missed=False, reward_weights=weights,
    )
    low_reward = compute_reward(
        previous_state=before, state=after,
        action=0, processed_priority="low", processed_delay_steps=3,
        deadline_missed=False, reward_weights=weights,
    )
    assert high_reward > medium_reward > low_reward


def test_compute_reward_penalizes_increasing_delay() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 3
    before.average_wait_steps = 10.0
    before.max_low_priority_wait_steps = 10
    before.max_high_wait = 10
    before.max_medium_wait = 10
    before.wait_steps_std = 2.0
    after_increase = SchedulerEnv().reset()
    after_increase.queue_length = 2
    after_increase.average_wait_steps = 15.0
    after_increase.max_low_priority_wait_steps = 15
    after_increase.max_high_wait = 15
    after_increase.max_medium_wait = 15
    after_increase.wait_steps_std = 3.0
    after_decrease = SchedulerEnv().reset()
    after_decrease.queue_length = 2
    after_decrease.average_wait_steps = 5.0
    after_decrease.max_low_priority_wait_steps = 5
    after_decrease.max_high_wait = 5
    after_decrease.max_medium_wait = 5
    after_decrease.wait_steps_std = 1.0

    weights = {
        "throughput_weight": 1.0,
        "delay_weight": 0.5,
        "fairness_weight": 0.1,
        "deadline_weight": 2.0,
    }
    reward_increase = compute_reward(
        previous_state=before, state=after_increase,
        action=0, processed_priority="medium", processed_delay_steps=3,
        deadline_missed=False, reward_weights=weights,
    )
    reward_decrease = compute_reward(
        previous_state=before, state=after_decrease,
        action=0, processed_priority="medium", processed_delay_steps=3,
        deadline_missed=False, reward_weights=weights,
    )
    assert reward_decrease > reward_increase


def test_compute_reward_rewards_improved_fairness() -> None:
    before = SchedulerEnv().reset()
    before.queue_length = 6
    before.average_wait_steps = 20.0
    before.max_low_priority_wait_steps = 80
    before.max_high_wait = 10
    before.max_medium_wait = 30
    before.wait_steps_std = 25.0
    after = SchedulerEnv().reset()
    after.queue_length = 5
    after.average_wait_steps = 15.0
    after.max_low_priority_wait_steps = 30
    after.max_high_wait = 20
    after.max_medium_wait = 25
    after.wait_steps_std = 3.0

    weights = {
        "throughput_weight": 0.0,
        "delay_weight": 0.1,
        "fairness_weight": 2.0,
        "deadline_weight": 2.0,
    }
    reward = compute_reward(
        previous_state=before, state=after,
        action=0, processed_priority="low", processed_delay_steps=5,
        deadline_missed=False, reward_weights=weights,
    )
    assert reward > 0


def test_queue_manager_deadline_count() -> None:
    qm = QueueManager()
    qm.push(CDCEvent("e1", "high", arrival_step=0, sync_cost=1.0, deadline_step=5))
    qm.push(CDCEvent("e2", "low", arrival_step=1, sync_cost=1.0, deadline_step=3))
    qm.push(CDCEvent("e3", "medium", arrival_step=2, sync_cost=1.0, deadline_step=None))
    assert qm.deadline_missed_count() == 0

    qm.increment_wait_steps(4)
    assert qm.deadline_missed_count() == 1

    qm.increment_wait_steps(2)
    assert qm.deadline_missed_count() == 2

    event = qm.pop_priority("high")
    assert event is not None
    assert qm.deadline_missed_count() == 1

    qm.increment_wait_steps(0)
    assert qm.deadline_missed_count() == 1


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


def test_double_dqn_agent_can_optimize() -> None:
    agent = DoubleDQNAgent(
        action_count=6,
        state_dim=8,
        replay_capacity=16,
        batch_size=4,
        device="cpu",
    )

    for index in range(6):
        state = [float(index), 1.0, 2.0, 3.0, 4.0, 5.0, 0.2, 0.4]
        next_state = [float(index + 1), 1.5, 2.5, 3.5, 4.5, 5.5, 0.3, 0.5]
        agent.store_transition(
            state=state,
            action=index % 6,
            reward=1.0,
            next_state=next_state,
            done=index == 5,
        )

    loss = agent.optimize()

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


def test_run_scheduler_training_smoke(monkeypatch) -> None:
    temp_dir = _make_scheduler_temp_dir().resolve()
    try:
        project_root = temp_dir
        dataset_dir = project_root / "data" / "scheduler_processed"
        output_dir = project_root / "outputs" / "scheduler"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(1, 13)],
                "timestamp": [
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:02",
                    "2024-01-01 00:00:03",
                    "2024-01-01 00:00:04",
                    "2024-01-01 00:00:05",
                    "2024-01-01 00:00:06",
                    "2024-01-01 00:00:07",
                    "2024-01-01 00:00:08",
                    "2024-01-01 00:00:09",
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:00:11",
                    "2024-01-01 00:00:12",
                ],
                "priority_label": [
                    "high", "medium", "low",
                    "high", "medium", "low",
                    "high", "medium", "low",
                    "high", "medium", "low",
                ],
                "estimated_sync_cost": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "deadline": [5.0] * 12,
            }
        )
        frame.iloc[:8].to_csv(dataset_dir / "train.csv", index=False)
        frame.iloc[8:10].to_csv(dataset_dir / "valid.csv", index=False)
        frame.iloc[10:].to_csv(dataset_dir / "test.csv", index=False)

        config_path = project_root / "scheduler_smoke.yaml"
        config = {
            "dataset_config": str(project_root / "configs" / "dataset.yaml"),
            "scheduler_dataset_dir": str(dataset_dir),
            "classifier_output_dir": str(project_root / "outputs" / "classifier"),
            "algorithm": "dqn",
            "episode_count": 1,
            "max_steps_per_episode": 8,
            "train_event_window": 6,
            "valid_event_window": 2,
            "random_state": 42,
            "queue_capacity": 100,
            "starvation_threshold": 5,
            "defer_low_priority": True,
            "low_release_batch_size": 2,
            "low_release_load_threshold": 0.35,
            "low_release_high_queue_threshold": 2,
            "low_force_release_wait_steps": 10,
            "off_peak_start_hour": 0,
            "off_peak_end_hour": 6,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "batch_size": 4,
            "replay_capacity": 32,
            "target_update_interval": 4,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.99,
            "reward_weights": {
                "throughput_weight": 1.0,
                "delay_weight": 0.1,
                "fairness_weight": 0.5,
                "deferred_low_weight": 0.05,
                "off_peak_release_weight": 0.1,
                "deadline_weight": 2.0,
            },
            "output_dir": str(output_dir),
            "run_name": "pytest_scheduler_smoke",
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        monkeypatch.setattr(
            "cdc_priority.scheduler.training.default_settings",
            lambda: AppSettings(
                project_root=project_root,
                configs_dir=project_root / "configs",
                outputs_dir=project_root / "outputs",
            ),
        )

        resolved_output_dir = run_scheduler_training(config_path)

        report_path = output_dir / "pytest_scheduler_smoke" / "scheduler_report.json"
        comparison_path = output_dir / "pytest_scheduler_smoke" / "policy_comparison.csv"
        model_path = output_dir / "pytest_scheduler_smoke" / "dqn_agent.pt"

        assert report_path.exists()
        assert comparison_path.exists()
        assert model_path.exists()
        assert resolved_output_dir == output_dir / "pytest_scheduler_smoke"

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["algorithm"] == "dqn"
        assert "dqn_test_metrics" in report
        assert "policy_comparison" in report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
