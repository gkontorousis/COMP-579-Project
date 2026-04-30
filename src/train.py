from pathlib import Path
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from src.registration import register_stock_envs
import gymnasium as gym


def main():
    # smoke test for ensuring the integration with SB3 baselines is working with our custom Gymnasium environment (stock_env.py)
    parser = argparse.ArgumentParser("SB3 smoke training for RLStock")
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_out", type=Path, default=Path("outputs/ddpg_smoke_model"))
    args = parser.parse_args()

    register_stock_envs(
        data_path=args.dj_30_dp_path,
        dji_path=args.dji_path,
        init_balance=10_000,
        max_shares_per_trade=5,
    )

    env = DummyVecEnv([lambda: gym.make("RLStockTrain-v0")])
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    print("Smoke train OK. Reward sample:", rewards)
    env.close()


if __name__ == "__main__":
    main()
