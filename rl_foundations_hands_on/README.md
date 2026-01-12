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
