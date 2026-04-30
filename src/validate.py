from pathlib import Path
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import gymnasium as gym
from src.registration import register_stock_envs


def validate(model_path, env):
    """Run one deterministic episode of a trained model; return cumulative reward."""
    model = DDPG.load(str(model_path), env=env)
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        total_reward += float(rewards[0])
        done = bool(dones[0])
    return total_reward


def main():
    parser = argparse.ArgumentParser("Validate trained DDPG on RLStockValidation-v0")
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    register_stock_envs(data_path=args.dj_30_dp_path)
    env = DummyVecEnv([lambda: gym.make("RLStockValidation-v0")])
    total_reward = validate(args.model_path, env)
    print(f"Validation total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
