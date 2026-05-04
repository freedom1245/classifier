from dataclasses import dataclass


@dataclass
class CDCEvent:
    event_id: str
    priority: str
    arrival_step: int
    sync_cost: float
    arrival_hour: int | None = None
    deadline_step: int | None = None
    wait_steps: int = 0
    service_steps: int = 1

    @property
    def priority_rank(self) -> int:
        ranks = {
            "low": 0,
            "medium": 1,
            "high": 2,
        }
        return ranks.get(self.priority, 0)
