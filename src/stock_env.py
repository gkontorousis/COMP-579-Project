from pathlib import Path
import argparse
from src import data_loader
from src import plots
from src import baseline_strategies as bstrat

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces


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
        self.prices_array = np.array(
            [day_df["adjcp"].values for day_df in self.daily_data], dtype=np.float64
        )
        self.mode = mode
        self.init_day = day
        self.init_balance = init_balance
        self.max_shares_per_trade = max_shares_per_trade

        if self.mode == "test":
            if dji_path is None:
                raise ValueError("dji_path must be provided when mode='test'.")

            episode_dates = bstrat.episode_dates_series(self.daily_data)
            self.dji_growth = bstrat.compute_dji_account_growth(
                path=dji_path,
                episode_dates=episode_dates,
                init_balance=self.init_balance,
            )
            self.min_variance_growth = bstrat.compute_min_variance_portfolio_growth(
                self.daily_data,
                episode_dates,
                self.init_balance,
            )

        self.day = day
        self.n_stocks = self.prices_array.shape[1]

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
            dtype=np.float32,
        )

        # [cash] + [prices] + [holdings]; size is 1 + n_stocks (prices) + n_stocks (holdings)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1 + 2 * self.n_stocks,), dtype=np.float32
        )

        self.state = (
            [init_balance]
            + self.prices_array[self.init_day].tolist()
            + [0 for i in range(self.n_stocks)]
        )

        self.reward = 0
        self.asset_memory = [init_balance]

        # Set by the caller (e.g. rl_algos.train / trade_online) to control where
        # the episode-result figure is written.  None → default names in cwd.
        self.figure_out: Path | None = None

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
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        actions = np.rint(actions).astype(np.int8)

        terminated = self.day >= self.terminal_day
        truncated = False

        if terminated:
            self.last_episode_asset_memory = list(self.asset_memory)
            plots.save_episode_result_figure(self, outfile=self.figure_out)

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

        self.state = (
            [self.state[self.__state_cash_idx]]
            + self.prices_array[self.day].tolist()
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
        self.state = (
            [self.init_balance]
            + self.prices_array[self.day].tolist()
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


def main():
    # smoke test to ensure custom environment runs as expected
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
        choices=["train", "test", "full"],
        help='environment mode; can be "train", "test", or "full" (runs both)',
        required=False,
    )
    args = args_parser.parse_args()
    data_path = args.dj_30_dp_path
    dji_path = args.dji_path
    mode = args.mode
    modes_to_run = ["train", "validation", "test"] if mode == "full" else [mode]
    for run_mode in modes_to_run:
        env = StockEnv(
            data_path=data_path,
            mode=run_mode,
            day=0,
            init_balance=10_000,
            max_shares_per_trade=5,
            dji_path=dji_path,
        )
        if run_mode == "test":
            assert hasattr(env, "dji_growth")
            assert hasattr(env, "min_variance_growth")

        # Reset determinism for same seed.
        obs_a, info_a = env.reset(seed=42)
        obs_b, info_b = env.reset(seed=42)
        assert np.allclose(obs_a, obs_b), f"non-deterministic reset for mode={run_mode}"
        assert isinstance(info_a, dict) and isinstance(info_b, dict)

        obs = obs_b
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

        max_steps = len(env.daily_data) + 2
        steps = 0
        done = False
        reward = 0.0
        while not done and steps < max_steps:
            if steps == 0:
                action = env.action_space.sample()
            elif steps == 1:
                action = env.action_space.sample().astype(np.float32) + 0.49
            elif steps == 2:
                action = env.action_space.sample().reshape(1, -1)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == env.observation_space.shape
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs)), f"non-finite obs at step {steps} mode={run_mode}"
            assert np.isfinite(reward), f"non-finite reward at step {steps} mode={run_mode}"
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)

            # State invariants: cash and holdings are non-negative.
            n = env.n_stocks
            cash = float(obs[0])
            prices = obs[1 : 1 + n]
            holdings = obs[1 + n :]
            assert cash >= 0.0, f"negative cash at step {steps} mode={run_mode}"
            assert np.all(holdings >= 0.0), f"negative holdings at step {steps} mode={run_mode}"
            assert np.all(np.isclose(holdings, np.round(holdings))), (
                f"non-integer holdings at step {steps} mode={run_mode}"
            )

            # Portfolio value from state should match latest recorded asset value.
            portfolio_value = float(
                cash + np.dot(prices.astype(np.float64), holdings.astype(np.float64))
            )
            assert np.isclose(env.asset_memory[-1], portfolio_value, atol=1e-5), (
                f"asset_memory mismatch at step {steps} mode={run_mode}"
            )

            done = bool(terminated or truncated)
            steps += 1

        # should eventually finish
        assert done, f"episode did not terminate within {max_steps} steps for mode={run_mode}"
        env.close()
        print(f"Smoke test passed for mode={run_mode}. steps={steps}, final_reward={reward:.4f}")


# run command
# uv run python -m src.stock_env --dj_30_dp_path data/dow_jones_30_daily_price.csv --dji_path 'data/^DJI.csv' --mode full
if __name__ == "__main__":
    main()
