"""Q-learning on delayed-reward env (optimizes long-term return)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from envs.delayed_reward_env import DelayedRewardEnv

def train_q_learning(episodes: int = 5000, alpha: float = 0.2, gamma: float = 0.95,
                     epsilon: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    env0 = DelayedRewardEnv(seed=seed)
    n_states = env0.max_fatigue + 1
    n_actions = 2
    Q = np.zeros((n_states, n_actions), dtype=float)

    for ep in range(episodes):
        env = DelayedRewardEnv(seed=seed + ep)
        s = env.reset()
        done = False
        while not done:
            if rng.random() < epsilon:
                a = int(rng.integers(0, n_actions))
            else:
                a = int(np.argmax(Q[s]))

            s2, r, done = env.step(a)
            td_target = r + (0.0 if done else gamma * np.max(Q[s2]))
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s2

        if (ep + 1) % 500 == 0:
            epsilon = max(0.05, epsilon * 0.95)

    return Q

def evaluate(policy_fn, episodes: int = 300, seed: int = 999):
    returns = []
    for ep in range(episodes):
        env = DelayedRewardEnv(seed=seed + ep)
        s = env.reset()
        total = 0.0
        done = False
        while not done:
            a = int(policy_fn(s))
            s, r, done = env.step(a)
            total += r
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))

def main():
    Q = train_q_learning()
    policy = lambda s: int(np.argmax(Q[s]))

    mean_ret, std_ret = evaluate(policy)
    print(f"Q-learning long-term return: mean={mean_ret:.2f}, std={std_ret:.2f}")
    print("Greedy action by fatigue state (0..10):", [int(np.argmax(Q[s])) for s in range(Q.shape[0])])

    mean_cb, _ = evaluate(lambda s: 0)
    mean_q, _  = evaluate(lambda s: 1)
    print(f"Always clickbait return: {mean_cb:.2f}")
    print(f"Always quality return  : {mean_q:.2f}")

if __name__ == "__main__":
    main()
