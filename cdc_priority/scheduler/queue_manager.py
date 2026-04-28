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
    wait_offset: int = 0
    size: int = 0
    relative_wait_sum: float = 0.0

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
        # 事件内部只保存“相对等待值”，全局等待增长统一由 wait_offset 维护。
        event.wait_steps -= self.wait_offset
        queue = self.priority_queues.setdefault(event.priority, deque())
        queue.append(event)
        self.size += 1
        self.relative_wait_sum += event.wait_steps

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
        return event

    def pop_priority(self, priority: str) -> CDCEvent | None:
        return self._pop_from_priority(priority)

    def peek_priority(self, priority: str) -> CDCEvent | None:
        queue = self.priority_queues.get(priority)
        if not queue:
            return None
        return queue[0]

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

    def priority_counts(self) -> dict[str, int]:
        return {
            priority: len(queue)
            for priority, queue in self.priority_queues.items()
        }

    def average_wait_steps(self) -> float:
        if self.size == 0:
            return 0.0
        return (self.relative_wait_sum + self.wait_offset * self.size) / self.size

    def max_wait_steps_for_priority(self, priority: str) -> int:
        queue = self.priority_queues.get(priority)
        if not queue:
            return 0
        # 同一优先级队列按到达顺序入队，队首等待时间最大。
        return self.effective_wait_steps(queue[0])

    def __len__(self) -> int:
        return self.size
