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
