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
