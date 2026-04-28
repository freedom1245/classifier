from dataclasses import dataclass, field

from .event import CDCEvent
from .policies import (
    ACTION_NAMES,
    aging_light_policy,
    aging_strong_policy,
    fifo_policy,
    high_guarded_aging_policy,
    low_rescue_policy,
    strict_priority_policy,
)
from .queue_manager import QueueManager
from .reward import compute_reward


@dataclass
class SchedulerState:
    queue_length: int
    average_wait_steps: float
    priority_counts: dict[str, int]
    max_low_priority_wait_steps: int
    low_queue_fraction: float
    starvation_pressure: float
    max_high_wait: int
    max_medium_wait: int
    deadline_missed_count: int
    wait_steps_std: float
    remaining_fraction: float
    avg_sync_cost: float

    def to_vector(self) -> list[float]:
        return [
            float(self.queue_length),
            float(self.average_wait_steps),
            float(self.priority_counts.get("high", 0)),
            float(self.priority_counts.get("medium", 0)),
            float(self.priority_counts.get("low", 0)),
            float(self.max_low_priority_wait_steps),
            float(self.low_queue_fraction),
            float(self.starvation_pressure),
            float(self.max_high_wait),
            float(self.max_medium_wait),
            float(self.deadline_missed_count),
            float(self.wait_steps_std),
            float(self.remaining_fraction),
            float(self.avg_sync_cost),
        ]

    def fairness_index(self) -> float:
        waits = [
            float(self.max_high_wait),
            float(self.max_medium_wait),
            float(self.max_low_priority_wait_steps),
        ]
        numerator = sum(waits) ** 2
        denominator = 3 * sum(w * w for w in waits)
        if denominator < 1e-12:
            return 1.0
        return numerator / denominator

    def allowed_actions(self, starvation_threshold: int) -> list[int]:
        high_count = int(self.priority_counts.get("high", 0))
        low_count = int(self.priority_counts.get("low", 0))
        pressure = float(self.starvation_pressure)
        allowed = {0, 1, 2, 3, 4, 5}

        # 没有低优待处理时，不需要做低优救援。
        if low_count <= 0:
            allowed.discard(5)

        # 高优存在但低优压力还不够强时，禁止最激进的低优优先动作。
        if high_count > 0 and pressure < 2.0:
            allowed.discard(5)

        # 高优存在且低优压力还不算特别严重时，避免直接上强 aging。
        if high_count > 0 and pressure < 1.5:
            allowed.discard(3)

        # 高低优同时积压时，保留 guarded 动作为优先折中选项。
        if high_count > 0 and low_count > 0 and pressure >= 1.0:
            allowed.add(4)

        if not allowed:
            return list(range(len(SchedulerEnv.ACTION_NAMES)))
        return sorted(allowed)


@dataclass
class SchedulerEnv:
    events: list[CDCEvent] = field(default_factory=list)
    queue_manager: QueueManager = field(default_factory=QueueManager)
    reward_weights: dict[str, float] = field(default_factory=dict)
    starvation_threshold: int = 5
    current_step: int = 0
    next_event_index: int = 0
    total_events: int = 0

    ACTION_NAMES = ACTION_NAMES

    def reset(self) -> SchedulerState:
        self.queue_manager = QueueManager()
        self.current_step = 0
        self.next_event_index = 0
        self.total_events = len(self.events)
        self._enqueue_arrivals()
        return self.get_state()

    def _enqueue_arrivals(self) -> None:
        while (
            self.next_event_index < len(self.events)
            and self.events[self.next_event_index].arrival_step <= self.current_step
        ):
            source = self.events[self.next_event_index]
            self.queue_manager.push(
                CDCEvent(
                    event_id=source.event_id,
                    priority=source.priority,
                    arrival_step=source.arrival_step,
                    sync_cost=source.sync_cost,
                    deadline_step=source.deadline_step,
                    wait_steps=0,
                    service_steps=source.service_steps,
                )
            )
            self.next_event_index += 1

    def _select_event(self, action: int) -> CDCEvent | None:
        # 细粒度动作空间：
        # 0 fifo
        # 1 strict_priority
        # 2 aging_light
        # 3 aging_strong
        # 4 high_guarded_aging
        # 5 low_rescue
        if action == 0:
            return fifo_policy(self.queue_manager)
        if action == 1:
            return strict_priority_policy(self.queue_manager)
        if action == 2:
            return aging_light_policy(
                self.queue_manager,
                starvation_threshold=self.starvation_threshold,
            )
        if action == 3:
            return aging_strong_policy(
                self.queue_manager,
                starvation_threshold=self.starvation_threshold,
            )
        if action == 4:
            return high_guarded_aging_policy(
                self.queue_manager,
                starvation_threshold=self.starvation_threshold,
            )
        if action == 5:
            return low_rescue_policy(
                self.queue_manager,
                starvation_threshold=self.starvation_threshold,
            )
        raise ValueError(f"Unsupported action: {action}")

    def get_state(self) -> SchedulerState:
        priority_counts = self.queue_manager.priority_counts()
        queue_length = len(self.queue_manager)
        max_low_priority_wait_steps = self.queue_manager.max_wait_steps_for_priority("low")
        remaining_fraction = (
            (self.total_events - self.next_event_index) / max(self.total_events, 1)
        )
        return SchedulerState(
            queue_length=queue_length,
            average_wait_steps=self.queue_manager.average_wait_steps(),
            priority_counts=priority_counts,
            max_low_priority_wait_steps=max_low_priority_wait_steps,
            low_queue_fraction=priority_counts.get("low", 0) / max(queue_length, 1),
            starvation_pressure=max_low_priority_wait_steps / max(self.starvation_threshold, 1),
            max_high_wait=self.queue_manager.max_wait_steps_for_priority("high"),
            max_medium_wait=self.queue_manager.max_wait_steps_for_priority("medium"),
            deadline_missed_count=self.queue_manager.deadline_missed_count(),
            wait_steps_std=self.queue_manager.wait_steps_std(),
            remaining_fraction=remaining_fraction,
            avg_sync_cost=self.queue_manager.avg_sync_cost(),
        )

    def step(self, action: int) -> tuple[SchedulerState, float, bool, dict]:
        if len(self.queue_manager) == 0 and self.next_event_index < len(self.events):
            self.current_step = max(
                self.current_step,
                self.events[self.next_event_index].arrival_step,
            )
            self._enqueue_arrivals()

        previous_state = self.get_state()
        processed_event = self._select_event(action) if len(self.queue_manager) > 0 else None
        processed_delay_steps = 0
        deadline_missed = False
        info = {"action": action, "processed_event_id": None, "processed_priority": None}

        if processed_event is not None:
            processed_delay_steps = max(self.current_step - processed_event.arrival_step, 0)
            if processed_event.deadline_step is not None:
                deadline_missed = processed_delay_steps > processed_event.deadline_step
            info["processed_event_id"] = processed_event.event_id
            info["processed_priority"] = processed_event.priority
            # 处理一条事件后，全局等待偏移和仿真时间同步前进。
            self.queue_manager.increment_wait_steps(processed_event.service_steps)
            self.current_step += processed_event.service_steps
        else:
            self.current_step += 1

        self._enqueue_arrivals()
        state = self.get_state()
        reward = compute_reward(
            previous_state=previous_state,
            state=state,
            action=action,
            processed_priority=info["processed_priority"],
            processed_delay_steps=processed_delay_steps,
            deadline_missed=deadline_missed,
            reward_weights=self.reward_weights,
        )
        done = self.next_event_index >= len(self.events) and len(self.queue_manager) == 0
        info["processed_delay_steps"] = processed_delay_steps
        info["deadline_missed"] = deadline_missed
        return state, reward, done, info
