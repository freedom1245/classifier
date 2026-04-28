from collections import deque
from dataclasses import dataclass, field
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical


class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_count: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOPolicyValueNetwork(nn.Module):
    def __init__(self, state_dim: int, action_count: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_count)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


@dataclass
class DQNAgent:
    action_count: int
    state_dim: int
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    replay_capacity: int = 5000
    batch_size: int = 64
    learning_rate: float = 1e-3
    device: str = "cpu"
    policy_network: DQNetwork = field(init=False)
    target_network: DQNetwork = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    replay_buffer: deque = field(init=False)

    def __post_init__(self) -> None:
        self.policy_network = DQNetwork(self.state_dim, self.action_count).to(self.device)
        self.target_network = DQNetwork(self.state_dim, self.action_count).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
        )
        self.replay_buffer = deque(maxlen=self.replay_capacity)

    def select_action(self, state_vector: list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_count)
        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_values = self.policy_network(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self.replay_buffer.append((state, action, reward, next_state, done))

    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_network(states_tensor).gather(1, actions_tensor).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states_tensor).max(dim=1).values
            target_q = rewards_tensor + self.gamma * next_q * (1.0 - dones_tensor)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


@dataclass
class PPOAgent:
    action_count: int
    state_dim: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    mini_batch_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    device: str = "cpu"
    policy_value_network: PPOPolicyValueNetwork = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    rollout_buffer: list[dict[str, float | int | list[float] | bool]] = field(init=False)

    def __post_init__(self) -> None:
        self.policy_value_network = PPOPolicyValueNetwork(
            self.state_dim,
            self.action_count,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_value_network.parameters(),
            lr=self.learning_rate,
        )
        self.rollout_buffer = []

    def select_action(
        self,
        state_vector: list[float],
        deterministic: bool = False,
        allowed_actions: list[int] | None = None,
    ) -> tuple[int, float, float]:
        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            logits, value = self.policy_value_network(state_tensor)
            if allowed_actions is not None and len(allowed_actions) < self.action_count:
                masked_logits = torch.full_like(logits, -1e9)
                masked_logits[:, allowed_actions] = logits[:, allowed_actions]
                logits = masked_logits
            distribution = Categorical(logits=logits)
            action_tensor = (
                logits.argmax(dim=1)
                if deterministic
                else distribution.sample()
            )
            log_prob = distribution.log_prob(action_tensor)
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def estimate_value(self, state_vector: list[float]) -> float:
        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            _, value = self.policy_value_network(state_tensor)
        return float(value.item())

    def store_transition(
        self,
        state: list[float],
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.rollout_buffer.append(
            {
                "state": state,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "done": done,
                "value": value,
            }
        )

    def optimize(self, last_value: float = 0.0) -> float:
        if not self.rollout_buffer:
            return 0.0

        rewards = [float(item["reward"]) for item in self.rollout_buffer]
        dones = [bool(item["done"]) for item in self.rollout_buffer]
        values = [float(item["value"]) for item in self.rollout_buffer]
        states = [list(item["state"]) for item in self.rollout_buffer]
        actions = [int(item["action"]) for item in self.rollout_buffer]
        old_log_probs = [float(item["log_prob"]) for item in self.rollout_buffer]

        advantages: list[float] = []
        gae = 0.0
        next_value = last_value
        for index in reversed(range(len(rewards))):
            mask = 0.0 if dones[index] else 1.0
            delta = rewards[index] + self.gamma * next_value * mask - values[index]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[index]

        returns = [adv + value for adv, value in zip(advantages, values, strict=True)]

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if len(advantages) > 1:
            advantages_tensor = (
                advantages_tensor - advantages_tensor.mean()
            ) / (advantages_tensor.std(unbiased=False) + 1e-8)

        total_loss = 0.0
        update_count = 0
        sample_count = len(self.rollout_buffer)
        for _ in range(self.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.mini_batch_size):
                batch_indices = permutation[start : start + self.mini_batch_size]
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                logits, state_values = self.policy_value_network(batch_states)
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(state_values, batch_returns)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.policy_value_network.parameters(),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                total_loss += float(loss.item())
                update_count += 1

        self.rollout_buffer.clear()
        return total_loss / max(update_count, 1)
