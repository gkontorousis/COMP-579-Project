"""Full pipeline: tune → train (with best params) → trade (online learning)."""

from pathlib import Path
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from src.registration import register_stock_envs
from src import rl_algos


def main():
    parser = argparse.ArgumentParser(
        "RLStock full pipeline: tune → train → trade", allow_abbrev=False
    )
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_out",
        type=Path,
        default=None,
        help="Checkpoint path (default: outputs/<algo>_model)",
    )
    parser.add_argument("--n_trials", type=int, default=5, help="Optuna trials (0 to skip tuning)")
    parser.add_argument(
        "--timesteps_per_trial", type=int, default=3_000, help="Training timesteps per tuning trial"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ddpg",
        choices=rl_algos.SUPPORTED_ALGO_NAMES,
        help="Stable-Baselines3 off-policy actor-critic algorithm",
    )
    args = parser.parse_args()
    algo_cls = rl_algos.get_algo_class(args.algo)
    if args.model_out is None:
        args.model_out = Path("outputs") / f"{args.algo}_model"

    register_stock_envs(
        data_path=args.dj_30_dp_path,
        dji_path=args.dji_path,
        init_balance=10_000,
        max_shares_per_trade=5,
    )

    # --- tune: find best hyperparams (each trial: train + validate) ---
    if args.n_trials > 0:
        print(f"Tuning over {args.n_trials} trials ({args.timesteps_per_trial} steps each) ...")
        best_params = rl_algos.tune(
            algo_cls,
            "RLStockTrain-v0",
            "RLStockValidation-v0",
            args.timesteps_per_trial,
            args.seed,
            args.n_trials,
        )
        print("Best params:", best_params)
    else:
        best_params = {}

    # --- train: full training with best params ---
    sigma = best_params.pop("noise_sigma", 0.1)
    train_env = DummyVecEnv([lambda: gym.make("RLStockTrain-v0")])
    print(f"Training for {args.timesteps} timesteps ...")
    rl_algos.train(
        algo_cls, train_env, args.timesteps, args.seed, args.model_out, sigma=sigma, **best_params
    )
    train_env.close()
    print(f"Model saved to {args.model_out}")

    # --- trade: test episode with continued online training ---
    online_model_out = args.model_out.parent / (args.model_out.name + "_online")
    test_env = DummyVecEnv([lambda: gym.make("RLStockTest-v0")])
    print("Trading (online learning on test episode) ...")
    rl_algos.trade_online(algo_cls, args.model_out, test_env, online_model_out)
    test_env.close()

    print("\n--- Summary ---")
    print(f"  Algorithm         : {args.algo.upper()}")
    if args.n_trials > 0:
        print(f"  Best tuned params : {best_params}, noise_sigma={sigma:.3f}")
    print(f"  Model             : {args.model_out}")
    print(f"  Online model      : {online_model_out}")
    print("  Result figure     : result_test.png")


if __name__ == "__main__":
    main()
