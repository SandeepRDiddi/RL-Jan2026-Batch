"""A tiny delayed-reward environment to show why RL differs from supervised.

- action 0 = "clickbait": higher immediate reward, increases fatigue a lot
- action 1 = "quality"  : moderate reward, reduces fatigue

State: fatigue (0..max_fatigue)
Reward: base - fatigue_penalty
Goal: maximize long-term return over the horizon
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class DelayedRewardEnv:
    max_fatigue: int = 10
    horizon: int = 50
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.fatigue = 0
        return int(self.fatigue)

    def step(self, action: int):
        assert action in (0, 1), "action must be 0 (clickbait) or 1 (quality)"

        base = 1.0 if action == 0 else 0.6
        reward = max(0.0, base - 0.06 * self.fatigue)

        if action == 0:
            self.fatigue = min(self.max_fatigue, self.fatigue + 2)
        else:
            self.fatigue = max(0, self.fatigue - 1)

        self.t += 1
        done = self.t >= self.horizon
        return int(self.fatigue), float(reward), done
