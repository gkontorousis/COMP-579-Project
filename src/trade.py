from pathlib import Path
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from src.registration import register_stock_envs


def trade(model_path, env):
    """Run one deterministic test episode; the env saves the result figure on termination.
    Returns cumulative reward."""
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
    parser = argparse.ArgumentParser("Trade/test trained DDPG on RLStockTest-v0")
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    args = parser.parse_args()

    register_stock_envs(data_path=args.dj_30_dp_path, dji_path=args.dji_path)
    env = DummyVecEnv([lambda: gym.make("RLStockTest-v0")])
    total_reward = trade(args.model_path, env)
    print(f"Test total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
