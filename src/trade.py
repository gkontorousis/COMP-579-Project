from pathlib import Path
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from src.config import RunConfig
from src.registration import ORIGINAL_TRADE, register_stock_envs
from src import rl_algos


def trade(algo, model_path, env, model_out, figure_out=None):
    """Continue training on the test env for one episode (online learning) and save updated model."""
    rl_algos.trade_online(algo, model_path, env, model_out, figure_out=figure_out)


def main():
    parser = argparse.ArgumentParser(
        f"Trade/test with online learning on {ORIGINAL_TRADE}", allow_abbrev=False
    )
    parser.add_argument("--dj_30_dp_path", type=Path, required=True)
    parser.add_argument("--dji_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--algo",
        type=str,
        default="ddpg",
        choices=rl_algos.SUPPORTED_ALGO_NAMES,
        help="Algorithm that was used to train the model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_out",
        type=Path,
        default=None,
        help="Output path for online-updated model (default: <model_path>_online)",
    )
    args = parser.parse_args()

    algo_cls = rl_algos.get_algo_class(args.algo)
    model_out = args.model_out or (args.model_path.parent / (args.model_path.name + "_online"))
    cfg = RunConfig(algo_name=args.algo, seed=args.seed, model_out=model_out)
    cfg.mkdir()

    register_stock_envs(data_path=args.dj_30_dp_path, dji_path=args.dji_path)
    env = DummyVecEnv([lambda: gym.make(ORIGINAL_TRADE)])
    trade(algo_cls, args.model_path, env, cfg.online_model_out, figure_out=cfg.test_figure_out)
    env.close()
    print(f"Online-updated model saved to {cfg.online_model_out}")
    print(f"Test figure saved to       {cfg.test_figure_out}")


if __name__ == "__main__":
    main()
