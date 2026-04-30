"""Full pipeline: tune → train (with best params) → trade (online learning)."""

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from src.config import RunConfig
from src.data_loader import build_daily_frames
from src.registration import register_stock_envs
from src import rl_algos
from src.baseline_strategies import compute_portfolio_metrics


def _aggregate_metrics(per_seed: list[dict]) -> dict:
    """Mean ± std for the agent (seed-dependent); single value for deterministic baselines."""
    fields = list(per_seed[0]["metrics"]["agent"].keys())
    agg = {"agent": {}}
    for field in fields:
        vals = [r["metrics"]["agent"][field] for r in per_seed]
        agg["agent"][f"mean_{field}"] = round(float(np.mean(vals)), 6)
        agg["agent"][f"std_{field}"] = round(
            float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 6
        )

    for strategy in ("djia", "min_variance"):
        agg[strategy] = per_seed[0]["metrics"][strategy]

    return agg


def main():
    parser = argparse.ArgumentParser(
        "RLStock full pipeline: tune → train → trade", allow_abbrev=False
    )
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    ts_group = parser.add_mutually_exclusive_group()
    ts_group.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training environment steps (default: 10 000 if --n_episodes not given)",
    )
    ts_group.add_argument(
        "--n_episodes",
        type=int,
        default=None,
        help="Total training episodes; converted to timesteps using the train-split length",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=42, help="Single seed (default: 42)")
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="S",
        help="Multiple seeds: tune once with seeds[0], train+trade per seed",
    )
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

    seeds = args.seeds if args.seeds is not None else [args.seed]
    multi_seed = len(seeds) > 1

    if args.n_episodes is not None:
        n_train_days = len(build_daily_frames(pd.read_csv(args.dj_30_dp_path), mode="train"))
        timesteps = args.n_episodes * n_train_days
        timesteps_per_trial = timesteps
        print(f"--n_episodes {args.n_episodes} × {n_train_days} train days = {timesteps} timesteps")
    else:
        timesteps = args.timesteps if args.timesteps is not None else 10_000
        timesteps_per_trial = args.timesteps_per_trial

    # create run-root dir (used for aggregate JSON in multi-seed runs)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    register_stock_envs(
        data_path=args.dj_30_dp_path,
        dji_path=args.dji_path,
        init_balance=10_000,
        max_shares_per_trade=5,
    )

    # we only tune once with seeds[0] to find the best hyperparameters
    if args.n_trials > 0:
        print(
            f"Tuning over {args.n_trials} trials ({timesteps_per_trial} steps each) using seed {seeds[0]} ..."
        )
        best_params = rl_algos.tune(
            algo_cls,
            "RLStockTrain-v0",
            "RLStockValidation-v0",
            timesteps_per_trial,
            seeds[0],
            args.n_trials,
        )
        print("Best params:", best_params)
    else:
        best_params = {}

    sigma = best_params.pop("noise_sigma", 0.1)
    hyperparams = {**best_params, "noise_sigma": sigma}

    per_seed_results = []
    for seed in seeds:
        if multi_seed:
            seed_model_out = args.model_out.parent / f"seed_{seed}" / args.model_out.name
        else:
            seed_model_out = args.model_out

        cfg = RunConfig(algo_name=args.algo, seed=seed, model_out=seed_model_out)
        cfg.mkdir()

        print(f"\n[seed={seed}] Training for {timesteps} timesteps ...")
        train_env = DummyVecEnv([lambda: gym.make("RLStockTrain-v0")])
        rl_algos.train(
            algo_cls,
            train_env,
            timesteps,
            seed,
            cfg.model_out,
            sigma=sigma,
            figure_out=cfg.train_figure_out,
            **best_params,
        )
        train_env.close()

        print(f"[seed={seed}] Trading (online learning on test episode) ...")
        test_env = DummyVecEnv([lambda: gym.make("RLStockTest-v0")])
        rl_algos.trade_online(
            algo_cls,
            cfg.model_out,
            test_env,
            cfg.online_model_out,
            figure_out=cfg.test_figure_out,
        )

        inner = test_env.envs[0].unwrapped
        metrics = {
            "agent": compute_portfolio_metrics(np.array(inner.last_episode_asset_memory)),
            "djia": compute_portfolio_metrics(inner.dji_growth),
            "min_variance": compute_portfolio_metrics(inner.min_variance_growth),
        }
        test_env.close()

        result = {"seed": seed, "metrics": metrics}
        per_seed_results.append(result)

        report = {
            "run": {
                "algo": cfg.algo_name,
                "seed": seed,
                "model_out": str(cfg.model_out),
                "timesteps": timesteps,
                **({"n_episodes": args.n_episodes} if args.n_episodes is not None else {}),
            },
            "best_hyperparams": hyperparams,
            "metrics": metrics,
        }
        cfg.metrics_out.write_text(json.dumps(report, indent=2))
        print(f"[seed={seed}] Metrics JSON: {cfg.metrics_out}")

    if multi_seed:
        aggregate = {
            "run": {
                "algo": args.algo,
                "seeds": seeds,
                "timesteps": timesteps,
                **({"n_episodes": args.n_episodes} if args.n_episodes is not None else {}),
            },
            "best_hyperparams": hyperparams,
            "per_seed": per_seed_results,
            "aggregate": _aggregate_metrics(per_seed_results),
        }
        agg_path = args.model_out.parent / "aggregate_metrics.json"
        agg_path.write_text(json.dumps(aggregate, indent=2))
        print(f"\nAggregate metrics: {agg_path}")

    print("\n--- Summary ---")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Seeds     : {seeds}")
    if args.n_trials > 0:
        print(f"  Hyperparams (tuned with seed {seeds[0]}): {hyperparams}")


# uv run python -m src.main \
#   --dj_30_dp_path data/dow_jones_30_daily_price.csv \
#   --dji_path 'data/^DJI.csv' \
#   --algo ddpg \
#   --seeds 42 43 44 \
#   --n_episodes 50 \
#   --n_trials 10 \
#   --model_out outputs/ddpg_run1/ddpg_model
if __name__ == "__main__":
    main()
