import json
from pathlib import Path
import random

import torch

from ..settings import default_settings, load_yaml_config
from ..utils import ensure_directory
from .agent import DQNAgent, PPOAgent
from .env import SchedulerEnv
from .evaluate import (
    append_policy_result,
    export_policy_comparison,
    export_policy_comparison_figure,
    load_scheduler_events,
)
from .event import CDCEvent


ACTION_COUNT = len(SchedulerEnv.ACTION_NAMES)


def _slice_events(
    events: list[CDCEvent],
    start_index: int,
    window_size: int,
) -> list[CDCEvent]:
    subset = events[start_index : start_index + window_size]
    if not subset:
        return []
    arrival_base = subset[0].arrival_step
    return [
        CDCEvent(
            event_id=event.event_id,
            priority=event.priority,
            arrival_step=event.arrival_step - arrival_base,
            sync_cost=event.sync_cost,
            deadline_step=event.deadline_step,
            wait_steps=0,
            service_steps=event.service_steps,
        )
        for event in subset
    ]


def _sample_training_events(
    events: list[CDCEvent],
    window_size: int,
    rng: random.Random,
) -> list[CDCEvent]:
    if len(events) <= window_size:
        return _slice_events(events, 0, len(events))
    start_index = rng.randint(0, len(events) - window_size)
    return _slice_events(events, start_index, window_size)


def _validation_events(events: list[CDCEvent], window_size: int) -> list[CDCEvent]:
    if len(events) <= window_size:
        return _slice_events(events, 0, len(events))
    start_index = max((len(events) - window_size) // 2, 0)
    return _slice_events(events, start_index, window_size)


def _run_single_dqn_episode(
    env: SchedulerEnv,
    agent: DQNAgent,
    max_steps: int,
    target_update_interval: int,
) -> tuple[float, int, float]:
    state = env.reset()
    total_reward = 0.0
    losses: list[float] = []

    for step in range(max_steps):
        state_vector = state.to_vector()
        action = agent.select_action(state_vector)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(
            state_vector,
            action,
            reward,
            next_state.to_vector(),
            done,
        )
        loss = agent.optimize()
        if loss is not None:
            losses.append(loss)
        total_reward += reward
        state = next_state

        if (step + 1) % target_update_interval == 0:
            agent.update_target_network()
        if done:
            return total_reward, step + 1, sum(losses) / max(len(losses), 1)

    return total_reward, max_steps, sum(losses) / max(len(losses), 1)


def _run_single_ppo_episode(
    env: SchedulerEnv,
    agent: PPOAgent,
    max_steps: int,
) -> tuple[float, int, float]:
    state = env.reset()
    total_reward = 0.0
    steps_used = 0

    for step in range(max_steps):
        state_vector = state.to_vector()
        action, log_prob, value = agent.select_action(
            state_vector,
            deterministic=False,
            allowed_actions=state.allowed_actions(env.starvation_threshold),
        )
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(
            state_vector,
            action,
            log_prob,
            reward,
            done,
            value,
        )
        total_reward += reward
        state = next_state
        steps_used = step + 1
        if done:
            break

    last_value = 0.0
    if steps_used > 0 and not done:
        last_value = agent.estimate_value(state.to_vector())
    loss = agent.optimize(last_value=last_value)
    return total_reward, steps_used, loss


def _evaluate_deterministic_policy(
    env: SchedulerEnv,
    action_selector,
    max_steps: int,
) -> float:
    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = action_selector(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def _evaluate_validation_summary(
    env: SchedulerEnv,
    action_selector,
    max_steps: int,
) -> dict[str, float]:
    state = env.reset()
    total_reward = 0.0
    delay_totals: list[int] = []
    high_delay_totals: list[int] = []
    per_priority_delay: dict[str, list[int]] = {"high": [], "medium": [], "low": []}

    for _ in range(max_steps):
        action = action_selector(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        priority = info.get("processed_priority")
        if priority is not None:
            delay = int(info.get("processed_delay_steps", 0))
            delay_totals.append(delay)
            per_priority_delay[priority].append(delay)
            if priority == "high":
                high_delay_totals.append(delay)
        if done:
            break

    average_delay_steps = sum(delay_totals) / max(len(delay_totals), 1)
    high_average_delay_steps = sum(high_delay_totals) / max(len(high_delay_totals), 1)
    per_class_means = [
        sum(per_priority_delay["high"]) / max(len(per_priority_delay["high"]), 1),
        sum(per_priority_delay["medium"]) / max(len(per_priority_delay["medium"]), 1),
        sum(per_priority_delay["low"]) / max(len(per_priority_delay["low"]), 1),
    ]
    numerator = sum(per_class_means) ** 2
    denominator = len(per_class_means) * sum(value * value for value in per_class_means)
    fairness_index = 0.0 if denominator <= 1e-12 else numerator / denominator

    return {
        "validation_reward": total_reward,
        "validation_average_delay_steps": average_delay_steps,
        "validation_high_priority_average_delay_steps": high_average_delay_steps,
        "validation_fairness_index": fairness_index,
    }


def _is_better_validation_candidate(
    candidate: dict[str, float],
    best: dict[str, float] | None,
    reward_tolerance: float,
    high_delay_tolerance: float,
    fairness_tolerance: float,
) -> bool:
    if best is None:
        return True

    reward_gap = candidate["validation_reward"] - best["validation_reward"]
    if reward_gap > reward_tolerance:
        return True
    if reward_gap < -reward_tolerance:
        return False

    high_delay_gap = (
        best["validation_high_priority_average_delay_steps"]
        - candidate["validation_high_priority_average_delay_steps"]
    )
    if high_delay_gap > high_delay_tolerance:
        return True
    if high_delay_gap < -high_delay_tolerance:
        return False

    fairness_gap = candidate["validation_fairness_index"] - best["validation_fairness_index"]
    if fairness_gap > fairness_tolerance:
        return True
    if fairness_gap < -fairness_tolerance:
        return False

    return (
        candidate["validation_average_delay_steps"]
        < best["validation_average_delay_steps"]
    )


def _evaluate_trained_agent(
    events: list[CDCEvent],
    reward_weights: dict[str, float],
    starvation_threshold: int,
    action_selector,
    policy_name: str,
) -> tuple[dict[str, float | int | str], dict[str, int]]:
    env = SchedulerEnv(
        events=events,
        reward_weights=reward_weights,
        starvation_threshold=starvation_threshold,
    )
    state = env.reset()
    delay_totals: list[int] = []
    high_delay_totals: list[int] = []
    per_priority_delay: dict[str, list[int]] = {"high": [], "medium": [], "low": []}
    action_counts = {
        action_name: 0
        for action_name in SchedulerEnv.ACTION_NAMES
    }
    completed = 0

    for _ in range(len(events) * 10):
        action = action_selector(state)
        action_counts[SchedulerEnv.ACTION_NAMES[action]] += 1
        state, _, done, info = env.step(action)
        priority = info.get("processed_priority")
        if priority is not None:
            delay = int(info.get("processed_delay_steps", 0))
            delay_totals.append(delay)
            per_priority_delay[priority].append(delay)
            if priority == "high":
                high_delay_totals.append(delay)
            completed += 1
        if done:
            break

    average_delay_steps = sum(delay_totals) / max(len(delay_totals), 1)
    high_average_delay_steps = sum(high_delay_totals) / max(len(high_delay_totals), 1)
    max_low_priority_wait_steps = max(per_priority_delay["low"], default=0)
    per_class_means = [
        sum(per_priority_delay["high"]) / max(len(per_priority_delay["high"]), 1),
        sum(per_priority_delay["medium"]) / max(len(per_priority_delay["medium"]), 1),
        sum(per_priority_delay["low"]) / max(len(per_priority_delay["low"]), 1),
    ]
    numerator = sum(per_class_means) ** 2
    denominator = len(per_class_means) * sum(value * value for value in per_class_means)
    fairness_index = 0.0 if denominator <= 1e-12 else numerator / denominator
    throughput = completed / max(env.current_step, 1)

    return (
        {
            "policy": policy_name,
            "throughput": throughput,
            "average_delay_steps": average_delay_steps,
            "high_priority_average_delay_steps": high_average_delay_steps,
            "max_low_priority_wait_steps": max_low_priority_wait_steps,
            "fairness_index": fairness_index,
            "completed_events": completed,
        },
        action_counts,
    )


def run_scheduler_training(config_path: Path) -> None:
    settings = default_settings()
    config = load_yaml_config(config_path)
    output_dir = ensure_directory(settings.project_root / config.values["output_dir"])
    dataset_dir = settings.project_root / config.values["scheduler_dataset_dir"]
    train_events = load_scheduler_events(dataset_dir / "train.csv")
    valid_events = load_scheduler_events(dataset_dir / "valid.csv")
    test_events = load_scheduler_events(dataset_dir / "test.csv")

    algorithm = str(config.values.get("algorithm", "dqn")).lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_weights = dict(config.values.get("reward_weights", {}))
    starvation_threshold = int(config.values.get("starvation_threshold", 50))
    max_steps = int(config.values.get("max_steps_per_episode", 500))
    episode_count = int(config.values.get("episode_count", 200))
    target_update_interval = int(config.values.get("target_update_interval", 20))
    train_event_window = int(config.values.get("train_event_window", 12000))
    valid_event_window = int(config.values.get("valid_event_window", 6000))
    random_state = int(config.values.get("random_state", 42))
    selection_reward_tolerance = float(config.values.get("selection_reward_tolerance", 1000.0))
    selection_high_delay_tolerance = float(
        config.values.get("selection_high_delay_tolerance", 250.0)
    )
    selection_fairness_tolerance = float(
        config.values.get("selection_fairness_tolerance", 1e-4)
    )
    rng = random.Random(random_state)

    validation_events = _validation_events(valid_events, valid_event_window)
    valid_env = SchedulerEnv(
        events=validation_events,
        reward_weights=reward_weights,
        starvation_threshold=starvation_threshold,
    )
    state_dim = len(valid_env.reset().to_vector())

    if algorithm == "ppo":
        agent = PPOAgent(
            action_count=ACTION_COUNT,
            state_dim=state_dim,
            gamma=float(config.values.get("gamma", 0.99)),
            gae_lambda=float(config.values.get("gae_lambda", 0.95)),
            learning_rate=float(config.values.get("learning_rate", 3e-4)),
            clip_epsilon=float(config.values.get("ppo_clip_epsilon", 0.2)),
            update_epochs=int(config.values.get("ppo_update_epochs", 4)),
            mini_batch_size=int(config.values.get("ppo_mini_batch_size", 128)),
            value_coef=float(config.values.get("ppo_value_coef", 0.5)),
            entropy_coef=float(config.values.get("ppo_entropy_coef", 0.01)),
            max_grad_norm=float(config.values.get("ppo_max_grad_norm", 0.5)),
            hidden_dim=int(config.values.get("ppo_hidden_dim", 128)),
            device=device,
        )
    else:
        agent = DQNAgent(
            action_count=ACTION_COUNT,
            state_dim=state_dim,
            gamma=float(config.values.get("gamma", 0.99)),
            epsilon=float(config.values.get("epsilon_start", 1.0)),
            epsilon_end=float(config.values.get("epsilon_end", 0.05)),
            epsilon_decay=float(config.values.get("epsilon_decay", 0.995)),
            replay_capacity=int(config.values.get("replay_capacity", 5000)),
            batch_size=int(config.values.get("batch_size", 64)),
            learning_rate=float(config.values.get("learning_rate", 1e-3)),
            device=device,
        )

    print(f"[scheduler] config: {config.path}")
    print(f"[scheduler] algorithm: {algorithm}")
    print(f"[scheduler] output_dir: {output_dir}")
    print(f"[scheduler] device: {device}")
    print(f"[scheduler] train events: {len(train_events)}")
    print(f"[scheduler] valid events: {len(valid_events)}")
    print(f"[scheduler] train window: {train_event_window}")
    print(f"[scheduler] valid window: {valid_event_window}")

    history: list[dict[str, float | int]] = []
    best_reward = float("-inf")
    best_state = None
    best_validation_summary: dict[str, float] | None = None

    for episode in range(episode_count):
        episode_events = _sample_training_events(train_events, train_event_window, rng)
        train_env = SchedulerEnv(
            events=episode_events,
            reward_weights=reward_weights,
            starvation_threshold=starvation_threshold,
        )

        if algorithm == "ppo":
            train_reward, steps_used, average_loss = _run_single_ppo_episode(
                train_env,
                agent,
                max_steps=max_steps,
            )
            validation_summary = _evaluate_validation_summary(
                valid_env,
                lambda state: agent.select_action(
                    state.to_vector(),
                    deterministic=True,
                    allowed_actions=state.allowed_actions(valid_env.starvation_threshold),
                )[0],
                max_steps=max_steps,
            )
            validation_reward = validation_summary["validation_reward"]
            best_candidate = {
                key: value.detach().cpu().clone()
                for key, value in agent.policy_value_network.state_dict().items()
            }
        else:
            train_reward, steps_used, average_loss = _run_single_dqn_episode(
                train_env,
                agent,
                max_steps=max_steps,
                target_update_interval=target_update_interval,
            )
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            validation_summary = _evaluate_validation_summary(
                valid_env,
                lambda state: agent.select_action(state.to_vector()),
                max_steps=max_steps,
            )
            validation_reward = validation_summary["validation_reward"]
            agent.epsilon = original_epsilon
            best_candidate = {
                key: value.detach().cpu().clone()
                for key, value in agent.policy_network.state_dict().items()
            }

        history.append(
            {
                "episode": episode + 1,
                "train_reward": train_reward,
                "validation_reward": validation_reward,
                "validation_high_priority_average_delay_steps": validation_summary[
                    "validation_high_priority_average_delay_steps"
                ],
                "validation_average_delay_steps": validation_summary[
                    "validation_average_delay_steps"
                ],
                "validation_fairness_index": validation_summary[
                    "validation_fairness_index"
                ],
                "steps_used": steps_used,
                "average_loss": average_loss,
                "epsilon": getattr(agent, "epsilon", 0.0),
            }
        )
        print(
            f"episode {episode + 1}/{episode_count} "
            f"train_reward={train_reward:.4f} "
            f"validation_reward={validation_reward:.4f} "
            f"validation_high_delay={validation_summary['validation_high_priority_average_delay_steps']:.4f} "
            f"validation_fairness={validation_summary['validation_fairness_index']:.4f} "
            f"control={getattr(agent, 'epsilon', 0.0):.4f}"
        )

        if _is_better_validation_candidate(
            validation_summary,
            best_validation_summary,
            reward_tolerance=selection_reward_tolerance,
            high_delay_tolerance=selection_high_delay_tolerance,
            fairness_tolerance=selection_fairness_tolerance,
        ):
            best_reward = validation_reward
            best_validation_summary = dict(validation_summary)
            best_state = best_candidate

        if algorithm == "dqn":
            agent.decay_epsilon()

    if best_state is not None:
        if algorithm == "ppo":
            agent.policy_value_network.load_state_dict(best_state)
        else:
            agent.policy_network.load_state_dict(best_state)
            agent.update_target_network()

    model_filename = "ppo_agent.pt" if algorithm == "ppo" else "dqn_agent.pt"
    model_path = output_dir / model_filename
    if algorithm == "ppo":
        torch.save(
            {
                "model_state_dict": agent.policy_value_network.state_dict(),
                "history": history,
                "best_validation_reward": best_reward,
            },
            model_path,
        )
        action_selector = lambda state_vector: agent.select_action(
            state_vector.to_vector(),
            deterministic=True,
            allowed_actions=state_vector.allowed_actions(starvation_threshold),
        )[0]
        trained_policy_name = "ppo"
    else:
        torch.save(
            {
                "model_state_dict": agent.policy_network.state_dict(),
                "history": history,
                "best_validation_reward": best_reward,
            },
            model_path,
        )
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action_selector = lambda state: agent.select_action(state.to_vector())
        trained_policy_name = "dqn"

    policy_comparison = export_policy_comparison(
        data_path=dataset_dir / "test.csv",
        output_path=output_dir / "policy_comparison.csv",
        starvation_threshold=starvation_threshold,
    )
    trained_metrics, action_counts = _evaluate_trained_agent(
        events=test_events,
        reward_weights=reward_weights,
        starvation_threshold=starvation_threshold,
        action_selector=action_selector,
        policy_name=trained_policy_name,
    )
    updated_comparison = append_policy_result(
        output_dir / "policy_comparison.csv",
        trained_metrics,
    )
    comparison_figure_path = export_policy_comparison_figure(
        comparison_csv_path=output_dir / "policy_comparison.csv",
        output_path=output_dir / "policy_comparison.png",
    )
    if algorithm == "dqn":
        agent.epsilon = original_epsilon

    report_path = output_dir / "scheduler_report.json"
    report_path.write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "best_validation_reward": best_reward,
                "best_validation_summary": best_validation_summary,
                "history": history,
                "train_event_window": train_event_window,
                "valid_event_window": valid_event_window,
                "policy_comparison": updated_comparison.to_dict(orient="records"),
                f"{trained_policy_name}_test_metrics": trained_metrics,
                f"{trained_policy_name}_action_counts": action_counts,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[scheduler] saved agent to: {model_path}")
    print(f"[scheduler] saved report to: {report_path}")
    print(f"[scheduler] saved comparison figure to: {comparison_figure_path}")
