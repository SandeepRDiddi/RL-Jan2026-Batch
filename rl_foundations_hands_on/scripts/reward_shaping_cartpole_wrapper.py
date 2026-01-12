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
