def compute_reward(
    previous_state,
    state,
    action: int,
    processed_priority: str | None,
    processed_delay_steps: int,
    deadline_missed: bool,
    reward_weights: dict[str, float],
) -> float:
    throughput = 0.0
    if processed_priority is not None:
        priority_multiplier = {"high": 2.0, "medium": 1.2, "low": 0.8}
        throughput = reward_weights.get(
            "throughput_weight", 1.0
        ) * priority_multiplier.get(processed_priority, 0.0)

    delay_delta = state.average_wait_steps - previous_state.average_wait_steps
    delay_reward = -reward_weights.get("delay_weight", 0.1) * delay_delta

    fairness_delta = state.fairness_index() - previous_state.fairness_index()
    fairness_reward = reward_weights.get("fairness_weight", 0.5) * fairness_delta

    deadline_penalty = -reward_weights.get("deadline_weight", 2.0) if deadline_missed else 0.0

    return throughput + delay_reward + fairness_reward + deadline_penalty
