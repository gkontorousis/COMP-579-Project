from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from src import rl_algos
from src.baseline_strategies import aggregate_metrics, compute_portfolio_metrics
from src.config import RunConfig
from src.data_loader import build_daily_frames
from src.registration import COVID_TRADE, COVID_TRAIN, register_covid_stock_envs


# Set these from your original-run best trial per algorithm.
# obtained from looking at selected_outputs/final/<model>/aggregate_metrics.json
BEST_HYPERPARAMS = {
    "ddpg": {
        "learning_rate": 0.000392989895488213,
        "batch_size": 128,
        "gamma": 0.9953442161638132,
        "tau": 0.0035289346993698885,
        "noise_sigma": 0.29866673511524433,
    },
    "td3": {
        "learning_rate": 0.00023457068305639934,
        "batch_size": 256,
        "gamma": 0.9524590591309414,
        "tau": 0.009332042579760615,
        "noise_sigma": 0.27322016716107367,
    },
}


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def run_one_seed(task):
    seed = int(task["seed"])
    algo_name = task["algo"]
    algo_cls = rl_algos.get_algo_class(algo_name)
    cfg = RunConfig(algo_name=algo_name, seed=seed, model_out=Path(task["seed_model_out"]))
    cfg.mkdir()

    best_params = dict(BEST_HYPERPARAMS[algo_name])
    sigma = best_params.pop("noise_sigma", 0.1)

    register_covid_stock_envs(
        train_data_path=task["covid_train_dp_path"],
        trade_data_path=task["covid_trade_dp_path"],
        dji_path=task["covid_dji_path"],
        init_balance=10_000,
        max_shares_per_trade=5,
    )

    log(f"[seed={seed}] train start (timesteps={task['timesteps']})")
    train_env = DummyVecEnv([lambda: gym.make(COVID_TRAIN)])
    rl_algos.train(
        algo_cls,
        train_env,
        task["timesteps"],
        seed,
        cfg.model_out,
        sigma=sigma,
        figure_out=cfg.train_figure_out,
        **best_params,
    )
    train_env.close()

    log(f"[seed={seed}] trade start")
    test_env = DummyVecEnv([lambda: gym.make(COVID_TRADE)])
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

    report = {
        "run": {
            "algo": cfg.algo_name,
            "seed": seed,
            "model_out": str(cfg.model_out),
            "timesteps": task["timesteps"],
            **({"n_episodes": task["n_episodes"]} if task["n_episodes"] is not None else {}),
        },
        "metrics": metrics,
    }
    cfg.metrics_out.write_text(json.dumps(report, indent=2))
    log(f"[seed={seed}] done; metrics={cfg.metrics_out}")
    return {"seed": seed, "metrics": metrics}


def main():
    run_start = time.time()
    parser = argparse.ArgumentParser("COVID run: train then online trade", allow_abbrev=False)
    parser.add_argument("--covid_train_dp_path", type=Path, required=True)
    parser.add_argument("--covid_trade_dp_path", type=Path, required=True)
    parser.add_argument("--covid_dji_path", type=Path, required=True)

    parser.add_argument("--n_episodes", type=int, default=10)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=42)
    seed_group.add_argument("--seeds", type=int, nargs="+", metavar="S")
    parser.add_argument("--seed_workers", type=int, default=None)
    parser.add_argument("--model_out", type=Path, default=None)
    parser.add_argument(
        "--algo",
        type=str,
        default="ddpg",
        choices=rl_algos.SUPPORTED_ALGO_NAMES,
    )
    args = parser.parse_args()

    if args.model_out is None:
        args.model_out = Path("outputs") / f"{args.algo}_covid_model"
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    n_train_days = len(
        build_daily_frames(pd.read_csv(args.covid_train_dp_path), mode="train", from_yfinance=True)
    )
    timesteps = args.n_episodes * n_train_days
    log(
        f"algo={args.algo} n_episodes={args.n_episodes} n_train_days={n_train_days} timesteps={timesteps}",
    )

    seeds = args.seeds if args.seeds is not None else [args.seed]
    multi_seed = len(seeds) > 1
    tasks = []
    for seed in seeds:
        if multi_seed:
            seed_model_out = args.model_out.parent / f"seed_{seed}" / args.model_out.name
        else:
            seed_model_out = args.model_out
        tasks.append(
            {
                "algo": args.algo,
                "seed": seed,
                "seed_model_out": str(seed_model_out),
                "timesteps": timesteps,
                "n_episodes": args.n_episodes,
                "covid_train_dp_path": str(args.covid_train_dp_path),
                "covid_trade_dp_path": str(args.covid_trade_dp_path),
                "covid_dji_path": str(args.covid_dji_path),
            }
        )

    if multi_seed:
        workers = args.seed_workers if args.seed_workers is not None else len(seeds)
        workers = max(1, min(workers, len(seeds)))
        log(f"[mp] start seeds={len(seeds)} workers={workers}")
        per_seed_results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_one_seed, task) for task in tasks]
            done = 0
            for f in as_completed(futures):
                per_seed_results.append(f.result())
                done += 1
                log(f"[mp] completed {done}/{len(seeds)} seeds")
        per_seed_results.sort(key=lambda x: x["seed"])
    else:
        per_seed_results = [run_one_seed(tasks[0])]

    if multi_seed:
        aggregate = {
            "run": {
                "algo": args.algo,
                "seeds": seeds,
                "timesteps": timesteps,
                "n_episodes": args.n_episodes,
            },
            "per_seed": per_seed_results,
            "aggregate": aggregate_metrics(per_seed_results),
        }
        agg_path = args.model_out.parent / "aggregate_metrics.json"
        agg_path.write_text(json.dumps(aggregate, indent=2))
        log(f"[aggregate] wrote {agg_path}")

    log(f"Elapsed: {time.time() - run_start:.1f}s")


# run smoke-test (minimal time to compute) to check everything works good -- do not want to train for hours just for an error after :(
# uv run python -m src.covid_run --covid_train_dp_path data/covid/train/20100101_20191231_dow_jones_30_daily_price.csv --covid_trade_dp_path data/covid/test/20200101_20201231_dow_jones_30_daily_price.csv --covid_dji_path 'data/covid/test/20200101_20201231_^DJI.csv' --n_episodes 1 --seed 42 --algo ddpg --model_out outputs/covid_ddpg_report_smoke_test/ddpg_model
# uv run python -m src.covid_run --covid_train_dp_path data/covid/train/20100101_20191231_dow_jones_30_daily_price.csv --covid_trade_dp_path data/covid/test/20200101_20201231_dow_jones_30_daily_price.csv --covid_dji_path 'data/covid/test/20200101_20201231_^DJI.csv' --n_episodes 1 --seed 42 --algo td3 --model_out outputs/covid_td3_report_smoke_test/td3_model
if __name__ == "__main__":
    main()
