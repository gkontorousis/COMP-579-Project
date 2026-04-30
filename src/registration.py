from __future__ import annotations

from pathlib import Path
import gymnasium as gym

from gymnasium.envs.registration import register
import argparse


def register_stock_envs(
    data_path: Path | str,
    dji_path: Path | str | None = None,
    init_balance: int = 10_000,
    max_shares_per_trade: int = 5,
) -> None:
    """
    Register train/test stock env IDs that both map to StockEnv,
    using mode-specific kwargs.
    """
    data_path = str(data_path)
    dji_path = None if dji_path is None else str(dji_path)

    if "RLStockTrain-v0" not in gym.envs.registry:
        register(
            id="RLStockTrain-v0",
            entry_point="src.stock_env:StockEnv",
            kwargs={
                "data_path": data_path,
                "mode": "train",
                "day": 0,
                "init_balance": init_balance,
                "max_shares_per_trade": max_shares_per_trade,
            },
        )

    if "RLStockTest-v0" not in gym.envs.registry:
        register(
            id="RLStockTest-v0",
            entry_point="src.stock_env:StockEnv",
            kwargs={
                "data_path": data_path,
                "mode": "test",
                "day": 0,
                "init_balance": init_balance,
                "max_shares_per_trade": max_shares_per_trade,
                "dji_path": dji_path,
            },
        )


def main():
    # smoke test to verify that no runtime errors occur
    args_parser = argparse.ArgumentParser(
        "Registration logic for adding custom stock trading environment to gymnasium"
    )
    args_parser.add_argument(
        "--dj_30_dp_path", type=Path, help="Path to dow_jones_30_daily_price.csv", required=True
    )
    args_parser.add_argument("--dji_path", type=Path, help="Path to ^DJI.csv", required=True)
    args_parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help='environment mode; can be "train" or "test"',
        required=False,
    )
    args = args_parser.parse_args()
    data_path = args.dj_30_dp_path
    dji_path = args.dji_path

    register_stock_envs(data_path=data_path, dji_path=dji_path)
    assert "RLStockTrain-v0" in gym.envs.registry
    assert "RLStockTest-v0" in gym.envs.registry
    train_env = gym.make("RLStockTrain-v0")
    test_env = gym.make("RLStockTest-v0")
    for env in (train_env, test_env):
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        env.close()
    print("Registration smoke test passed.")


if __name__ == "__main__":
    main()
