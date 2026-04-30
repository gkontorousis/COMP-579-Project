"""Full pipeline: train → validate → trade (test)."""
from pathlib import Path
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import gymnasium as gym

from src.registration import register_stock_envs
from src import train as train_mod
from src import validate as validate_mod
from src import trade as trade_mod


def main():
    parser = argparse.ArgumentParser("RLStock full pipeline: train → validate → trade")
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_out", type=Path, default=Path("outputs/ddpg_model"))
    args = parser.parse_args()

    register_stock_envs(
        data_path=args.dj_30_dp_path,
        dji_path=args.dji_path,
        init_balance=10_000,
        max_shares_per_trade=5,
    )

    # --- train ---
    train_env = DummyVecEnv([lambda: gym.make("RLStockTrain-v0")])
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )
    print(f"Training for {args.timesteps} timesteps ...")
    train_mod.train(DDPG, train_env, args.timesteps, args.seed, args.model_out, action_noise)
    train_env.close()
    print(f"Model saved to {args.model_out}")

    # --- validate ---
    val_env = DummyVecEnv([lambda: gym.make("RLStockValidation-v0")])
    val_reward = validate_mod.validate(args.model_out, val_env)
    val_env.close()
    print(f"Validation total reward: {val_reward:.2f}")

    # --- trade (test) ---
    test_env = DummyVecEnv([lambda: gym.make("RLStockTest-v0")])
    test_reward = trade_mod.trade(args.model_out, test_env)
    test_env.close()
    print(f"Test total reward: {test_reward:.2f}")

    print("\n--- Summary ---")
    print(f"  Validation reward : {val_reward:.2f}")
    print(f"  Test reward       : {test_reward:.2f}")
    print("  Result figure     : result_test.png")


if __name__ == "__main__":
    main()
