from pathlib import Path
import argparse
from src import data_loader

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces
import matplotlib.pyplot as plt


def _compute_dji_account_growth(
    path, episode_dates, init_balance, date_col="Date", price_col="Adj Close"
):
    dji = pd.read_csv(path)
    dji[date_col] = pd.to_datetime(dji[date_col], errors="coerce")
    dji = dji.dropna(subset=[date_col, price_col]).sort_values(date_col)

    px = dji.set_index(date_col)[price_col].astype(float)

    # env dates come from datadate YYYYMMDD ints -> datetime
    ep_dates = pd.to_datetime(episode_dates.astype(str), format="%Y%m%d", errors="coerce")
    ep_dates = pd.DatetimeIndex(ep_dates)

    aligned_px = px.reindex(ep_dates).ffill().bfill()
    daily_ret = aligned_px.pct_change().fillna(0.0)

    growth = (1.0 + daily_ret).cumprod() * float(init_balance)

    return growth.to_numpy()


class StockEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: Path,
        mode="train",
        day=0,
        init_balance=10_000,
        max_shares_per_trade=5,
        dji_path=None,
    ):
        # save initialization arg vals
        # TODO: Generalize creation of daily_data to be able to use other things than the build_daily_frames which right now
        # is loading the original (hardcoded) csv data
        self.daily_data = data_loader.build_daily_frames(pd.read_csv(data_path), mode)
        self.mode = mode
        self.init_day = day
        self.init_balance = init_balance
        self.max_shares_per_trade = max_shares_per_trade

        if self.mode == "test":
            if self.mode == "test" and dji_path is None:
                raise ValueError("dji_path must be provided when mode='test'.")

            episode_dates = pd.Series([day_df["datadate"].iloc[0] for day_df in self.daily_data])
            self.dji_growth = _compute_dji_account_growth(
                path=dji_path,
                episode_dates=episode_dates,
                init_balance=self.init_balance,
            )

        self.day = day
        self.data = self.daily_data[self.day]
        self.n_stocks = len(self.data)

        self.terminal_day = len(self.daily_data) - 1

        # need to store offset as shape of state is [cash, adjcp_1, adjcp_2, ..., adjcp_n_stocks, hold_1, hold_2, ..., hold_n_stocks]
        # allows for easier accessing of the desired value from state
        self.__state_cash_idx = 0
        self.__state_price_offset = 1
        self.__state_holdings_offset = self.__state_price_offset + self.n_stocks

        # buy or sell maximum 5 shares
        self.action_space = spaces.Box(
            low=-max_shares_per_trade,
            high=max_shares_per_trade,
            shape=(self.n_stocks,),
            dtype=np.int8,
        )

        # [cash] + [prices] + [holdings]; size is 1 + n_stocks (prices) + n_stocks (holdings)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1 + 2 * self.n_stocks,), dtype=np.float32
        )

        self.state = (
            [init_balance] + self.data.adjcp.values.tolist() + [0 for i in range(self.n_stocks)]
        )

        self.reward = 0
        self.asset_memory = [init_balance]

        self.reset()
        self._seed()

    def _total_portfolio_value(self) -> float:
        lo_p, hi_p = self.__state_price_offset, self.__state_price_offset + self.n_stocks
        lo_h, hi_h = self.__state_holdings_offset, self.__state_holdings_offset + self.n_stocks
        p = np.asarray(self.state[lo_p:hi_p], dtype=np.float64)
        h = np.asarray(self.state[lo_h:hi_h], dtype=np.float64)
        return float(self.state[self.__state_cash_idx] + np.dot(p, h))

    def _sell_stock(self, index, action):
        _action = min(abs(action), self.state[index + self.__state_holdings_offset])
        if self.state[index + self.__state_holdings_offset] > 0:
            self.state[self.__state_cash_idx] += (
                self.state[index + self.__state_price_offset] * _action
            )
            self.state[self.__state_holdings_offset + index] -= _action

    def _buy_stock(self, index, action):
        available_amount = (
            self.state[self.__state_cash_idx] // self.state[index + self.__state_price_offset]
        )
        _action = min(available_amount, action)
        self.state[self.__state_cash_idx] -= self.state[index + self.__state_price_offset] * _action
        self.state[index + self.__state_holdings_offset] += _action

    def step(self, actions):
        # explicitly convert the actions from any policy using this env to be in integer space
        actions = np.asarray(actions).reshape(self.n_stocks)
        actions = np.clip(actions, self.action_space.low, self.action_space.high).astype(np.int8)

        terminated = self.day >= self.terminal_day
        truncated = False

        if terminated:
            plt.plot(self.asset_memory, "r", label="agent")
            if self.mode == "test":
                plt.plot(self.dji_growth, label="dji_growth")

            plt.legend()
            plt.savefig("result_test.png" if self.mode == "test" else "result_training.png")
            plt.close()

            total_reward = self._total_portfolio_value() - self.init_balance
            print(f"total_reward: {total_reward}")

            obs = np.asarray(self.state, dtype=np.float32)
            info = {}
            return obs, float(self.reward), terminated, truncated, info

        begin_total_asset = self.state[self.__state_cash_idx] + sum(
            np.array(self.state[self.__state_price_offset : self.__state_holdings_offset])
            * np.array(self.state[self.__state_holdings_offset :])
        )

        sort_order = np.argsort(actions)

        num_sells = int((actions < 0).sum())
        sell_indices = sort_order[:num_sells]

        num_buys = int((actions > 0).sum())
        buy_indices = sort_order[::-1][:num_buys]

        for idx in sell_indices:
            self._sell_stock(idx, actions[idx])

        for idx in buy_indices:
            self._buy_stock(idx, actions[idx])

        self.day += 1
        self.data = self.daily_data[self.day]

        self.state = (
            [self.state[self.__state_cash_idx]]
            + self.data.adjcp.values.tolist()
            + list(self.state[self.__state_holdings_offset :])
        )
        end_total_asset = self.state[self.__state_cash_idx] + sum(
            np.array(self.state[self.__state_price_offset : self.__state_holdings_offset])
            * np.array(self.state[self.__state_holdings_offset :])
        )

        self.reward = end_total_asset - begin_total_asset
        self.asset_memory.append(end_total_asset)

        obs = np.asarray(self.state, dtype=np.float32)
        info = {}

        return obs, float(self.reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed(seed)

        self.asset_memory = [self.init_balance]
        self.day = self.init_day
        self.data = self.daily_data[self.day]
        self.state = (
            [self.init_balance]
            + self.data.adjcp.values.tolist()
            + [0 for i in range(self.n_stocks)]
        )

        self.reward = 0

        obs = np.asarray(self.state, dtype=np.float32)
        info = {}

        return obs, info

    def render(self, mode="human"):
        return np.asarray(self.state, dtype=np.float32)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# smoke test to ensure custom environment runs as expected
def main():
    args_parser = argparse.ArgumentParser(
        "Custom Stock Trading Environment for RL training for Liu paper"
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
    mode = args.mode
    env = StockEnv(
        data_path=data_path,
        mode=mode,
        day=0,
        init_balance=10_000,
        max_shares_per_trade=5,
        dji_path=dji_path,
    )

    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    max_steps = len(env.daily_data) + 2
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward), f"non-finite reward at step {steps}"
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        done = bool(terminated or truncated)
        steps += 1

    # should eventually finish
    assert done, f"episode did not terminate within {max_steps} steps"
    print(f"Smoke test passed. steps={steps}, final_reward={reward:.4f}")


if __name__ == "__main__":
    main()
