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
