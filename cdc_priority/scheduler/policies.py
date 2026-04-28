from .fairness import aging_priority_key
from .queue_manager import QueueManager


ACTION_NAMES = (
    "fifo",
    "strict_priority",
    "aging_light",
    "aging_strong",
    "high_guarded_aging",
    "low_rescue",
)


def fifo_policy(queue_manager: QueueManager):
    return queue_manager.pop_fifo()


def strict_priority_policy(queue_manager: QueueManager):
    return queue_manager.pop_strict_priority()


def weighted_round_robin_policy(queue_manager: QueueManager):
    return queue_manager.pop_fifo()


def aging_policy(
    queue_manager: QueueManager,
    starvation_threshold: int = 5,
    low_rescue_multiplier: int = 2,
    medium_rescue_multiplier: int = 2,
    guard_high: bool = False,
):
    return queue_manager.pop_aging(
        starvation_threshold=starvation_threshold,
        aging_key_fn=aging_priority_key,
        low_rescue_multiplier=low_rescue_multiplier,
        medium_rescue_multiplier=medium_rescue_multiplier,
        guard_high=guard_high,
    )


def aging_light_policy(queue_manager: QueueManager, starvation_threshold: int = 5):
    return aging_policy(
        queue_manager,
        starvation_threshold=starvation_threshold,
        low_rescue_multiplier=4,
        medium_rescue_multiplier=5,
        guard_high=False,
    )


def aging_strong_policy(queue_manager: QueueManager, starvation_threshold: int = 5):
    return aging_policy(
        queue_manager,
        starvation_threshold=starvation_threshold,
        low_rescue_multiplier=2,
        medium_rescue_multiplier=2,
        guard_high=False,
    )


def high_guarded_aging_policy(queue_manager: QueueManager, starvation_threshold: int = 5):
    return aging_policy(
        queue_manager,
        starvation_threshold=starvation_threshold,
        low_rescue_multiplier=2,
        medium_rescue_multiplier=3,
        guard_high=True,
    )


def low_rescue_policy(queue_manager: QueueManager, starvation_threshold: int = 5):
    low_queue = queue_manager.priority_queues.get("low")
    if low_queue:
        low_wait = queue_manager.effective_wait_steps(low_queue[0])
        if starvation_threshold > 0 and low_wait >= starvation_threshold:
            return queue_manager.pop_priority("low")
    return strict_priority_policy(queue_manager)
