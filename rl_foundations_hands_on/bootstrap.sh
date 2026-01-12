set -e

mkdir -p scripts envs

cat > requirements.txt <<'EOF'
numpy>=1.26
pandas>=2.0
matplotlib>=3.7
gymnasium[classic-control]>=0.29
stable-baselines3>=2.3
torch>=2.2
tensorboard>=2.14
scikit-learn>=1.3
EOF

cat > README.md <<'EOF'
# RL Foundations (Hands-on) — Session 1

## Setup (Conda)
conda create -n rltrain python=3.11 -y
conda activate rltrain
pip install -r requirements.txt

## Quick health check
python scripts/lab_check.py

## Run demos
python scripts/cartpole_random_agent.py
python scripts/bandit_epsilon_greedy.py
python scripts/train_supervised_policy.py
python scripts/train_q_learning.py
python scripts/ppo_cartpole_smoke.py
python scripts/reward_shaping_cartpole_wrapper.py
EOF

cat > scripts/lab_check.py <<'EOF'
import shutil, sys, time

def check_imports():
    import numpy  # noqa
    import gymnasium  # noqa
    import torch  # noqa
    import stable_baselines3  # noqa

def check_cartpole():
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    total = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    return total

def check_ppo():
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64)
    t0 = time.time()
    model.learn(total_timesteps=2000)
    dt = time.time() - t0
    env.close()
    return dt

def main():
    print("=== RL Lab Check ===")
    print("Python:", sys.version.split()[0])

    free_gb = shutil.disk_usage("/").free / (1024**3)
    print(f"Disk free (GB): {free_gb:.1f}")
    if free_gb < 10:
        raise SystemExit("FAIL: <10GB free disk")

    try:
        check_imports()
        print("Imports: PASS (numpy, gymnasium, torch, stable-baselines3)")
    except Exception as e:
        raise SystemExit(f"FAIL: Imports error: {e}")

    try:
        total = check_cartpole()
        print(f"CartPole: PASS (100 steps, total_reward={total:.1f})")
    except Exception as e:
        raise SystemExit(f"FAIL: CartPole error: {e}")

    try:
        dt = check_ppo()
        print(f"PPO smoke: PASS (2000 timesteps in {dt:.1f}s)")
    except Exception as e:
        raise SystemExit(f"FAIL: PPO error: {e}")

    print("PASS: Lab ready for Session 1")

if __name__ == "__main__":
    main()
EOF

cat > scripts/cartpole_random_agent.py <<'EOF'
"""RL loop demo: Agent–Environment–Reward–Episode using CartPole."""
import gymnasium as gym

def run(episodes: int = 5, seed: int = 42):
    env = gym.make("CartPole-v1")
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()  # random agent
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        print(f"Episode {ep}: steps={steps}, total_reward={total:.1f}")
    env.close()

if __name__ == "__main__":
    run()
EOF

cat > scripts/bandit_epsilon_greedy.py <<'EOF'
"""Trial-and-error: Exploration vs Exploitation using epsilon-greedy bandits."""
import numpy as np

def run(k: int = 5, steps: int = 5000, epsilon: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_means = rng.normal(0.0, 1.0, size=k)   # unknown to the agent
    q = np.zeros(k)                             # estimated value per action
    n = np.zeros(k, dtype=int)                  # pull counts
    rewards = []

    for t in range(1, steps + 1):
        if rng.random() < epsilon:
            a = int(rng.integers(0, k))         # explore
        else:
            a = int(np.argmax(q))               # exploit

        r = rng.normal(true_means[a], 1.0)      # stochastic reward
        rewards.append(r)

        n[a] += 1
        q[a] += (r - q[a]) / n[a]               # incremental mean

        if t in {10, 100, 1000, steps}:
            print(f"t={t:>5} | avg_reward={np.mean(rewards):.3f} | best_est={int(np.argmax(q))}")

    print("\nTrue means:", np.round(true_means, 3))
    print("Est means :", np.round(q, 3))
    print("Pulls     :", n)

if __name__ == "__main__":
    run()
EOF

cat > envs/delayed_reward_env.py <<'EOF'
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
EOF

cat > scripts/train_supervised_policy.py <<'EOF'
"""Supervised vs RL demo:
1) generate logged data from a random behavior policy
2) train supervised model to predict immediate reward proxy ("click")
3) deploy greedy policy and evaluate long-term return
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from envs.delayed_reward_env import DelayedRewardEnv

def generate_logs(n_episodes: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ep in range(n_episodes):
        env = DelayedRewardEnv(seed=seed + ep)
        s = env.reset()
        done = False
        while not done:
            a = int(rng.integers(0, 2))
            s2, r, done = env.step(a)
            click = 1 if r >= 0.5 else 0  # immediate reward proxy
            X.append([s, a])
            y.append(click)
            s = s2
    return np.array(X), np.array(y)

def train_supervised(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    return clf, auc

def greedy_action(clf, state: int):
    p0 = clf.predict_proba([[state, 0]])[0, 1]
    p1 = clf.predict_proba([[state, 1]])[0, 1]
    return 0 if p0 >= p1 else 1

def evaluate(policy_fn, episodes: int = 200, seed: int = 123):
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
    X, y = generate_logs()
    clf, auc = train_supervised(X, y)
    print(f"Supervised AUC (immediate click proxy): {auc:.3f}")

    mean_ret, std_ret = evaluate(lambda s: greedy_action(clf, s))
    print(f"Greedy(supervised) long-term return: mean={mean_ret:.2f}, std={std_ret:.2f}")

    mean_cb, _ = evaluate(lambda s: 0)
    mean_q, _  = evaluate(lambda s: 1)
    print(f"Always clickbait return: {mean_cb:.2f}")
    print(f"Always quality return  : {mean_q:.2f}")

if __name__ == "__main__":
    main()
EOF

cat > scripts/train_q_learning.py <<'EOF'
"""Q-learning on delayed-reward env (optimizes long-term return)."""
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
EOF

cat > scripts/ppo_cartpole_smoke.py <<'EOF'
"""PPO smoke run on CartPole-v1 (CPU)."""
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def run(total_timesteps: int = 5000):
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    dt = time.time() - t0
    env.close()
    print(f"PASS: PPO trained {total_timesteps} timesteps in {dt:.1f}s")

if __name__ == "__main__":
    run()
EOF

cat > scripts/reward_shaping_cartpole_wrapper.py <<'EOF'
"""Reward design demo (reward shaping) using a wrapper."""
import gymnasium as gym

class AnglePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty_scale: float = 0.05):
        super().__init__(env)
        self.penalty_scale = penalty_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pole_angle = float(obs[2])  # radians
        shaped = float(reward) - self.penalty_scale * abs(pole_angle)
        info["pole_angle"] = pole_angle
        info["shaped_reward"] = shaped
        return obs, shaped, terminated, truncated, info

def run_demo(steps: int = 200):
    env = AnglePenaltyWrapper(gym.make("CartPole-v1"), penalty_scale=0.05)
    obs, info = env.reset(seed=0)
    total = 0.0
    for _ in range(steps):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total += float(r)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"PASS: shaped reward demo ran {steps} steps, total_shaped_reward={total:.2f}")

if __name__ == "__main__":
    run_demo()
EOF

echo "DONE. Next:"
echo "  1) conda activate rltrain"
echo "  2) pip install -r requirements.txt"
echo "  3) python scripts/lab_check.py"
