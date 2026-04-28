def compute_reward(
    previous_state,
    state,
    action: int,
    processed_priority: str | None,
    processed_delay_steps: int,
    deadline_missed: bool,
    starvation_threshold: int,
    reward_weights: dict[str, float],
) -> float:
    fifo_actions = {0}
    strict_actions = {1}
    aging_actions = {2, 3, 4, 5}
    light_aging_actions = {2}
    strong_aging_actions = {3, 5}
    guarded_aging_actions = {4}
    low_rescue_actions = {5}

    reward = 0.0
    if processed_priority == "high":
        reward += reward_weights.get("high_priority_throughput", 1.0)
        reward -= reward_weights.get("high_priority_delay_penalty", 0.0) * processed_delay_steps
    elif processed_priority == "medium":
        reward += reward_weights.get("high_priority_throughput", 1.0) * 0.25

    reward -= reward_weights.get("average_delay_penalty", 0.1) * processed_delay_steps

    starvation_penalty = reward_weights.get("starvation_penalty", 1.0)
    aging_starvation_bonus = reward_weights.get("aging_starvation_bonus", 1.5)
    strict_under_starvation_penalty = reward_weights.get(
        "strict_under_starvation_penalty",
        2.0,
    )
    strict_high_pressure_bonus = reward_weights.get("strict_high_pressure_bonus", 1.0)
    fifo_mixed_pressure_penalty = reward_weights.get("fifo_mixed_pressure_penalty", 2.0)
    aging_balancing_bonus = reward_weights.get("aging_balancing_bonus", 1.0)
    low_priority_service_bonus = reward_weights.get("low_priority_service_bonus", 1.0)
    low_queue_reduction_bonus = reward_weights.get("low_queue_reduction_bonus", 0.75)
    high_backlog_service_bonus = reward_weights.get("high_backlog_service_bonus", 2.0)
    guarded_high_service_bonus = reward_weights.get("guarded_high_service_bonus", 1.5)
    guarded_mixed_bonus = reward_weights.get("guarded_mixed_bonus", 1.5)
    defer_high_penalty = reward_weights.get("defer_high_penalty", 2.0)
    light_aging_high_penalty = reward_weights.get("light_aging_high_penalty", 2.0)
    low_rescue_high_penalty = reward_weights.get("low_rescue_high_penalty", 3.0)
    low_rescue_emergency_bonus = reward_weights.get("low_rescue_emergency_bonus", 1.5)
    low_rescue_overuse_penalty = reward_weights.get("low_rescue_overuse_penalty", 2.0)
    starvation_threshold = max(starvation_threshold, 1)
    starvation_pressure = previous_state.max_low_priority_wait_steps / starvation_threshold
    low_queue_fraction = (
        previous_state.priority_counts.get("low", 0) / max(previous_state.queue_length, 1)
    )
    high_queue_fraction = (
        previous_state.priority_counts.get("high", 0) / max(previous_state.queue_length, 1)
    )
    starvation_scale = min(starvation_pressure, 3.0)
    starvation_active = (
        previous_state.priority_counts.get("low", 0) > 0
        and previous_state.max_low_priority_wait_steps >= starvation_threshold
    )
    mixed_priority_pressure = starvation_active and previous_state.priority_counts.get("high", 0) > 0
    aging_high_priority_penalty = reward_weights.get("aging_high_priority_penalty", 2.0)
    wait_increase = max(
        state.max_low_priority_wait_steps - previous_state.max_low_priority_wait_steps,
        0,
    )
    wait_reduction = max(
        previous_state.max_low_priority_wait_steps - state.max_low_priority_wait_steps,
        0,
    )
    reward -= starvation_penalty * (1.0 + low_queue_fraction) * wait_increase
    reward += starvation_penalty * 0.75 * wait_reduction

    if processed_priority == "low" and previous_state.max_low_priority_wait_steps > 0:
        reward += starvation_penalty * 0.25
        reward += low_priority_service_bonus * starvation_scale
    elif processed_priority == "high" and previous_state.priority_counts.get("high", 0) > 0:
        reward += high_backlog_service_bonus * (1.0 + high_queue_fraction)
        if action in guarded_aging_actions:
            reward += guarded_high_service_bonus * max(starvation_scale, 1.0)
        elif action in strict_actions and starvation_active:
            reward += strict_high_pressure_bonus * (1.0 + high_queue_fraction)

    if previous_state.priority_counts.get("low", 0) > state.priority_counts.get("low", 0):
        reward += low_queue_reduction_bonus * (1.0 + 0.5 * starvation_scale)

    if starvation_active and action in aging_actions:
        reward += aging_starvation_bonus * (1.0 + low_queue_fraction) * starvation_scale
        if mixed_priority_pressure:
            reward += aging_balancing_bonus * starvation_scale
        if action in strong_aging_actions:
            reward += 0.5 * aging_starvation_bonus
        if action in guarded_aging_actions and processed_priority == "high":
            reward += aging_balancing_bonus
            reward += guarded_mixed_bonus * (1.0 + high_queue_fraction)
    elif starvation_active and action in strict_actions:
        reward -= strict_under_starvation_penalty * (1.0 + low_queue_fraction) * starvation_scale
    elif mixed_priority_pressure and action in fifo_actions:
        reward -= fifo_mixed_pressure_penalty * starvation_scale

    # aging 负责防饿死，但高优已经明显积压时，继续绕开 high 需要付出更高代价，
    # 否则策略很容易塌缩成“只要有低优压力就一路 aging”。
    if (
        action in aging_actions
        and previous_state.priority_counts.get("high", 0) > 0
        and processed_priority != "high"
    ):
        reward -= aging_high_priority_penalty * high_queue_fraction * max(starvation_scale, 1.0)
        if action in light_aging_actions:
            reward -= light_aging_high_penalty * high_queue_fraction
        if action in low_rescue_actions:
            reward -= low_rescue_high_penalty * high_queue_fraction * max(starvation_scale, 1.0)

    if action in low_rescue_actions:
        if processed_priority == "low" and starvation_scale >= 2.0:
            reward += low_rescue_emergency_bonus * starvation_scale
        elif previous_state.priority_counts.get("high", 0) > 0 and processed_priority != "high":
            reward -= low_rescue_overuse_penalty * high_queue_fraction

    if processed_priority != "high" and previous_state.priority_counts.get("high", 0) > 0:
        reward -= defer_high_penalty * high_queue_fraction

    if deadline_missed:
        reward -= reward_weights.get("deadline_miss_penalty", 1.5)
    return reward
