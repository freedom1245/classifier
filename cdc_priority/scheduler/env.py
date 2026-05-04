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
    deferred_low_count: int
    max_deferred_low_wait_steps: int
    current_hour: int
    is_off_peak: float
    current_load: float

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
            float(self.deferred_low_count),
            float(self.max_deferred_low_wait_steps),
            float(self.current_hour),
            float(self.is_off_peak),
            float(self.current_load),
        ]

    def fairness_index(self) -> float:
        waits = [
            float(self.max_high_wait),
            float(self.max_medium_wait),
            float(self.max_low_priority_wait_steps),
        ]
        numerator = sum(waits) ** 2
        denominator = 3 * sum(wait * wait for wait in waits)
        if denominator < 1e-12:
            return 1.0
        return numerator / denominator

    def allowed_actions(self, starvation_threshold: int) -> list[int]:
        high_count = int(self.priority_counts.get("high", 0))
        low_count = int(self.priority_counts.get("low", 0))
        pressure = float(self.starvation_pressure)
        allowed = {0, 1, 2, 3, 4, 5}

        if low_count <= 0:
            allowed.discard(5)

        if high_count > 0 and pressure < 2.0:
            allowed.discard(5)

        if high_count > 0 and pressure < 1.5:
            allowed.discard(3)

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
    queue_capacity: int = 10000
    defer_low_priority: bool = True
    low_release_batch_size: int = 16
    low_release_load_threshold: float = 0.35
    low_release_high_queue_threshold: int = 8
    low_force_release_wait_steps: int = 600
    off_peak_start_hour: int = 0
    off_peak_end_hour: int = 6
    current_step: int = 0
    next_event_index: int = 0
    total_events: int = 0
    current_hour: int = 0

    ACTION_NAMES = ACTION_NAMES

    def reset(self) -> SchedulerState:
        self.queue_manager = QueueManager()
        self.current_step = 0
        self.next_event_index = 0
        self.total_events = len(self.events)
        self.current_hour = 0
        self._enqueue_arrivals()
        self._release_deferred_low_events()
        return self.get_state()

    def _is_off_peak_hour(self) -> bool:
        if self.off_peak_start_hour == self.off_peak_end_hour:
            return True
        if self.off_peak_start_hour < self.off_peak_end_hour:
            return self.off_peak_start_hour <= self.current_hour < self.off_peak_end_hour
        return self.current_hour >= self.off_peak_start_hour or self.current_hour < self.off_peak_end_hour

    def _enqueue_arrivals(self) -> None:
        while (
            self.next_event_index < len(self.events)
            and self.events[self.next_event_index].arrival_step <= self.current_step
        ):
            source = self.events[self.next_event_index]
            if source.arrival_hour is not None:
                self.current_hour = source.arrival_hour
            copied_event = CDCEvent(
                event_id=source.event_id,
                priority=source.priority,
                arrival_step=source.arrival_step,
                sync_cost=source.sync_cost,
                arrival_hour=source.arrival_hour,
                deadline_step=source.deadline_step,
                wait_steps=0,
                service_steps=source.service_steps,
            )
            if self.defer_low_priority and source.priority == "low":
                self.queue_manager.push_deferred_low(copied_event)
            else:
                self.queue_manager.push(copied_event)
            self.next_event_index += 1

    def _release_deferred_low_events(self) -> int:
        deferred_count = self.queue_manager.deferred_low_count()
        if deferred_count <= 0:
            return 0

        priority_counts = self.queue_manager.priority_counts()
        current_load = len(self.queue_manager) / max(self.queue_capacity, 1)
        high_count = priority_counts.get("high", 0)
        max_deferred_wait = self.queue_manager.max_deferred_low_wait_steps()
        force_release = max_deferred_wait >= self.low_force_release_wait_steps
        low_load_window = (
            self._is_off_peak_hour()
            and current_load <= self.low_release_load_threshold
            and high_count <= self.low_release_high_queue_threshold
        )

        if not force_release and not low_load_window:
            return 0

        batch_size = 1 if force_release and not low_load_window else self.low_release_batch_size
        return self.queue_manager.release_deferred_low(max_count=max(batch_size, 1))

    def _select_event(self, action: int) -> CDCEvent | None:
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
            deferred_low_count=self.queue_manager.deferred_low_count(),
            max_deferred_low_wait_steps=self.queue_manager.max_deferred_low_wait_steps(),
            current_hour=self.current_hour,
            is_off_peak=1.0 if self._is_off_peak_hour() else 0.0,
            current_load=queue_length / max(self.queue_capacity, 1),
        )

    def step(self, action: int) -> tuple[SchedulerState, float, bool, dict]:
        if len(self.queue_manager) == 0 and self.next_event_index < len(self.events):
            self.current_step = max(
                self.current_step,
                self.events[self.next_event_index].arrival_step,
            )
            self._enqueue_arrivals()
            self._release_deferred_low_events()

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
            self.queue_manager.increment_wait_steps(processed_event.service_steps)
            self.current_step += processed_event.service_steps
        else:
            self.queue_manager.increment_wait_steps(1)
            self.current_step += 1

        self._enqueue_arrivals()
        self._release_deferred_low_events()
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
        done = (
            self.next_event_index >= len(self.events)
            and len(self.queue_manager) == 0
            and self.queue_manager.deferred_low_count() == 0
        )
        info["processed_delay_steps"] = processed_delay_steps
        info["deadline_missed"] = deadline_missed
        return state, reward, done, info
