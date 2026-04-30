from pathlib import Path
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from src.registration import register_stock_envs
from src import rl_algos


def trade(algo, model_path, env, model_out):
    """Continue training on the test env for one episode (online learning) and save updated model."""
    rl_algos.trade_online(algo, model_path, env, model_out)


def main():
    parser = argparse.ArgumentParser("Trade/test with online learning on RLStockTest-v0")
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--model_out", type=Path, default=Path("outputs/ddpg_model_online"))
    args = parser.parse_args()

    register_stock_envs(data_path=args.dj_30_dp_path, dji_path=args.dji_path)
    env = DummyVecEnv([lambda: gym.make("RLStockTest-v0")])
    trade(DDPG, args.model_path, env, args.model_out)
    env.close()
    print(f"Online-updated model saved to {args.model_out}")


if __name__ == "__main__":
    main()
