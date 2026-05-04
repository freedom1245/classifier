import math
from collections import deque
from dataclasses import dataclass, field
from itertools import chain

from .event import CDCEvent


@dataclass
class QueueManager:
    priority_queues: dict[str, deque[CDCEvent]] = field(
        default_factory=lambda: {
            "high": deque(),
            "medium": deque(),
            "low": deque(),
        }
    )
    deferred_low_queue: deque[CDCEvent] = field(default_factory=deque)
    wait_offset: int = 0
    size: int = 0
    deferred_size: int = 0
    relative_wait_sum: float = 0.0
    _deadline_missed: int = 0

    @property
    def events(self) -> list[CDCEvent]:
        return list(
            chain(
                self.priority_queues["high"],
                self.priority_queues["medium"],
                self.priority_queues["low"],
            )
        )

    def push(self, event: CDCEvent) -> None:
        event.wait_steps -= self.wait_offset
        queue = self.priority_queues.setdefault(event.priority, deque())
        queue.append(event)
        self.size += 1
        self.relative_wait_sum += event.wait_steps
        if self._is_event_deadline_missed(event):
            self._deadline_missed += 1

    def push_deferred_low(self, event: CDCEvent) -> None:
        event.wait_steps -= self.wait_offset
        self.deferred_low_queue.append(event)
        self.deferred_size += 1

    def _is_event_deadline_missed(self, event: CDCEvent) -> bool:
        if event.deadline_step is None:
            return False
        return self.effective_wait_steps(event) > event.deadline_step

    def _sync_deadline_counter(self) -> None:
        self._deadline_missed = 0
        for queue in self.priority_queues.values():
            for event in queue:
                if self._is_event_deadline_missed(event):
                    self._deadline_missed += 1

    def effective_wait_steps(self, event: CDCEvent) -> int:
        return event.wait_steps + self.wait_offset

    def _pop_from_priority(self, priority: str) -> CDCEvent | None:
        queue = self.priority_queues.get(priority)
        if not queue:
            return None
        event = queue.popleft()
        self.size -= 1
        self.relative_wait_sum -= event.wait_steps
        event.wait_steps = self.effective_wait_steps(event)
        if self._is_event_deadline_missed(event):
            self._deadline_missed -= 1
        return event

    def pop_priority(self, priority: str) -> CDCEvent | None:
        return self._pop_from_priority(priority)

    def peek_priority(self, priority: str) -> CDCEvent | None:
        queue = self.priority_queues.get(priority)
        if not queue:
            return None
        return queue[0]

    def peek_deferred_low(self) -> CDCEvent | None:
        if not self.deferred_low_queue:
            return None
        return self.deferred_low_queue[0]

    def release_deferred_low(self, max_count: int) -> int:
        released = 0
        while self.deferred_low_queue and released < max_count:
            event = self.deferred_low_queue.popleft()
            self.deferred_size -= 1
            self.push(event)
            released += 1
        return released

    def pop_fifo(self) -> CDCEvent | None:
        candidates: list[tuple[int, str]] = []
        for priority, queue in self.priority_queues.items():
            if queue:
                candidates.append((queue[0].arrival_step, priority))
        if not candidates:
            return None
        _, priority = min(candidates, key=lambda item: item[0])
        return self._pop_from_priority(priority)

    def pop_strict_priority(self) -> CDCEvent | None:
        for priority in ("high", "medium", "low"):
            if self.priority_queues[priority]:
                return self._pop_from_priority(priority)
        return None

    def pop_aging(
        self,
        starvation_threshold: int,
        aging_key_fn,
        low_rescue_multiplier: int = 2,
        medium_rescue_multiplier: int = 2,
        guard_high: bool = False,
    ) -> CDCEvent | None:
        high_head = self.peek_priority("high")
        high_wait = self.effective_wait_steps(high_head) if high_head is not None else 0

        low_head = self.peek_priority("low")
        if low_head is not None:
            low_wait = self.effective_wait_steps(low_head)
            low_rescue_threshold = starvation_threshold * max(low_rescue_multiplier, 1)
            if starvation_threshold > 0 and low_wait >= low_rescue_threshold:
                if not guard_high or high_head is None or low_wait >= high_wait + starvation_threshold:
                    return self._pop_from_priority("low")

        medium_head = self.peek_priority("medium")
        if medium_head is not None:
            medium_wait = self.effective_wait_steps(medium_head)
            medium_rescue_threshold = starvation_threshold * max(medium_rescue_multiplier, 1)
            if starvation_threshold > 0 and medium_wait >= medium_rescue_threshold:
                if not guard_high or high_head is None or medium_wait >= high_wait + starvation_threshold:
                    return self._pop_from_priority("medium")

        candidates: list[tuple[tuple[int, int, int, int], str]] = []
        for priority, queue in self.priority_queues.items():
            if not queue:
                continue
            event = queue[0]
            candidates.append(
                (
                    aging_key_fn(
                        event,
                        starvation_threshold,
                        wait_steps=self.effective_wait_steps(event),
                    ),
                    priority,
                )
            )
        if not candidates:
            return None
        _, priority = max(candidates, key=lambda item: item[0])
        return self._pop_from_priority(priority)

    def increment_wait_steps(self, delta: int = 1) -> None:
        self.wait_offset += delta
        self._sync_deadline_counter()

    def priority_counts(self) -> dict[str, int]:
        return {
            priority: len(queue)
            for priority, queue in self.priority_queues.items()
        }

    def deferred_low_count(self) -> int:
        return self.deferred_size

    def max_deferred_low_wait_steps(self) -> int:
        if not self.deferred_low_queue:
            return 0
        return self.effective_wait_steps(self.deferred_low_queue[0])

    def average_wait_steps(self) -> float:
        if self.size == 0:
            return 0.0
        return (self.relative_wait_sum + self.wait_offset * self.size) / self.size

    def max_wait_steps_for_priority(self, priority: str) -> int:
        queue = self.priority_queues.get(priority)
        if not queue:
            return 0
        return self.effective_wait_steps(queue[0])

    def __len__(self) -> int:
        return self.size

    def deadline_missed_count(self) -> int:
        return self._deadline_missed

    def wait_steps_std(self) -> float:
        if self.size < 2:
            return 0.0
        raw = [self.effective_wait_steps(event) for event in self.events]
        mean = sum(raw) / len(raw)
        variance = sum((x - mean) ** 2 for x in raw) / len(raw)
        return math.sqrt(variance)

    def avg_sync_cost(self) -> float:
        if self.size == 0:
            return 0.0
        return sum(event.sync_cost for event in self.events) / self.size
