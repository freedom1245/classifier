from .event import CDCEvent
from .queue_manager import QueueManager


def apply_aging(
    event: CDCEvent,
    starvation_threshold: int,
    wait_steps: int | None = None,
) -> CDCEvent:
    aged_event = event
    effective_wait = aged_event.wait_steps if wait_steps is None else wait_steps
    if effective_wait >= starvation_threshold and aged_event.priority == "low":
        aged_event.priority = "medium"
    if effective_wait >= starvation_threshold * 2 and aged_event.priority == "medium":
        aged_event.priority = "high"
    return aged_event


def effective_priority_rank(event: CDCEvent, starvation_threshold: int, wait_steps: int | None = None) -> int:
    boosted_rank = event.priority_rank
    effective_wait = event.wait_steps if wait_steps is None else wait_steps
    if starvation_threshold > 0:
        # aging 不再只是“轻微提权”，而是随着等待时间分段增强，
        # 让低优事件在明显饿死时能够真正压过高优队列。
        boosted_rank += min(effective_wait // starvation_threshold, 3)
    return min(boosted_rank, 4)


def aging_priority_key(
    event: CDCEvent,
    starvation_threshold: int,
    wait_steps: int | None = None,
) -> tuple[int, int, int, int]:
    effective_wait = event.wait_steps if wait_steps is None else wait_steps
    boosted_rank = effective_priority_rank(event, starvation_threshold, wait_steps=effective_wait)
    starvation_level = 0
    if starvation_threshold > 0:
        starvation_level = min(effective_wait // starvation_threshold, 3)
    return (
        boosted_rank,
        starvation_level,
        event.priority_rank,
        effective_wait,
    )
