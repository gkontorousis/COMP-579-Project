from __future__ import annotations

from pathlib import Path
import gymnasium as gym

from gymnasium.envs.registration import register
import argparse

ORIGINAL_TRAIN = "RLStockTrain-v0"
ORIGINAL_VALIDATE = "RLStockValidation-v0"
ORIGINAL_TRADE = "RLStockTest-v0"

# we only use train and test split; we reuse the best hyperparameters found from the original paper's date
# range optuna hyperparameter run due to time constraints
COVID_TRAIN = "Covid_RLStockTrain"
COVID_TRADE = "Covid_RLStockTrade"


def register_stock_envs(
    data_path: Path | str,
    dji_path: Path | str | None = None,
    init_balance: int = 10_000,
    max_shares_per_trade: int = 5,
) -> None:
    data_path = str(data_path)
    dji_path = None if dji_path is None else str(dji_path)

    if ORIGINAL_TRAIN not in gym.envs.registry:
        register(
            id=ORIGINAL_TRAIN,
            entry_point="src.stock_env:StockEnv",
            kwargs={
                "data_path": data_path,
                "mode": "train",
                "day": 0,
                "init_balance": init_balance,
                "max_shares_per_trade": max_shares_per_trade,
            },
        )

    if ORIGINAL_VALIDATE not in gym.envs.registry:
        register(
            id=ORIGINAL_VALIDATE,
            entry_point="src.stock_env:StockEnv",
            kwargs={
                "data_path": data_path,
                "mode": "validation",
                "day": 0,
                "init_balance": init_balance,
                "max_shares_per_trade": max_shares_per_trade,
            },
        )

    if ORIGINAL_TRADE not in gym.envs.registry:
        register(
            id=ORIGINAL_TRADE,
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


def register_covid_stock_envs(
    train_data_path: Path | str,
    trade_data_path: Path | str,
    dji_path: Path | str | None = None,
    init_balance: int = 10_000,
    max_shares_per_trade: int = 5,
) -> None:
    train_data_path = str(train_data_path)
    trade_data_path = str(trade_data_path)
    dji_path = None if dji_path is None else str(dji_path)

    if COVID_TRAIN not in gym.envs.registry:
        register(
            id=COVID_TRAIN,
            entry_point="src.stock_env:StockEnv",
            kwargs={
                "data_path": train_data_path,
                "mode": "train",
                "day": 0,
                "init_balance": init_balance,
                "max_shares_per_trade": max_shares_per_trade,
                "from_yfinance": True,
            },
        )

    if COVID_TRADE not in gym.envs.registry:
        register(
            id=COVID_TRADE,
            entry_point="src.stock_env:StockEnv",
            kwargs={
                "data_path": trade_data_path,
                "mode": "test",
                "day": 0,
                "init_balance": init_balance,
                "max_shares_per_trade": max_shares_per_trade,
                "dji_path": dji_path,
                "from_yfinance": True,
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
    assert ORIGINAL_TRAIN in gym.envs.registry
    assert ORIGINAL_TRADE in gym.envs.registry
    train_env = gym.make(ORIGINAL_TRAIN)
    test_env = gym.make(ORIGINAL_TRADE)
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
