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
