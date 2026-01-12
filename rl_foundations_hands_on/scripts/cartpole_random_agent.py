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
