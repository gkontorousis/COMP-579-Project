from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

# Annualization / PyPortfolioOpt frequency convention (trading days per year).
MHR_FREQ = 252


def compute_portfolio_metrics(values, freq=MHR_FREQ) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values) - 1
    initial = float(values[0])
    final = float(values[-1])
    ann_return = (final / initial) ** (freq / n) - 1
    daily_rets = np.diff(values) / values[:-1]
    ann_std = float(np.std(daily_rets, ddof=1)) * np.sqrt(freq)
    sharpe = ann_return / ann_std if ann_std > 0 else float("nan")
    return {
        "initial_portfolio_value": round(initial, 4),
        "final_portfolio_value": round(final, 4),
        "annualized_return": round(ann_return, 6),
        "annualized_std": round(ann_std, 6),
        "sharpe_ratio": round(sharpe, 6),
    }


def episode_dates_series(daily_data: list[Any]) -> pd.Series:
    """One YYYYMMDD int per `daily_data` row, in episode order (same as env trading days)."""
    return pd.Series([day_df["datadate"].iloc[0] for day_df in daily_data])


def compute_dji_account_growth(
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


def convert_daily_data_dfs_to_long_df_prices(daily_data_dfs):
    # helper method to convert the list of daily_data dfs into a df that can be used with PyPortfolioOpt
    long_df = pd.concat(daily_data_dfs, ignore_index=True)
    long_df["date"] = pd.to_datetime(
        long_df["datadate"].astype(str), format="%Y%m%d", errors="coerce"
    )
    prices = long_df.pivot(index="date", columns="tic", values="adjcp").sort_index()
    prices = prices.astype(float)

    return prices


def compute_min_variance_portfolio_growth(daily_data_dfs, episode_dates, init_balance):
    """Buy-and-hold min-vol portfolio value on each episode date (like `_compute_dji_account_growth`).

    Same reindex/ffill/bfill convention as :func:`compute_dji_account_growth`.
    """
    prices = convert_daily_data_dfs_to_long_df_prices(daily_data_dfs)
    prices = prices.dropna(how="all", axis=0).dropna(how="all", axis=1)
    prices = prices.ffill().bfill()

    mu = expected_returns.mean_historical_return(prices, frequency=MHR_FREQ)
    S = risk_models.sample_cov(prices, frequency=MHR_FREQ)
    ef = EfficientFrontier(mu, S)
    ef.min_volatility()
    cleaned = ef.clean_weights()
    # TODO:
    w = (
        pd.Series({k: float(v) for k, v in cleaned.items()}, dtype=float)
        .reindex(prices.columns)
        .fillna(0.0)
    )
    s = float(w.sum())
    if s > 0:
        w = w / s

    asset_rets = prices.pct_change().fillna(0.0)
    port_ret = (asset_rets * w).sum(axis=1)

    ep_dates = pd.to_datetime(episode_dates.astype(str), format="%Y%m%d", errors="coerce")
    ep_dates = pd.DatetimeIndex(ep_dates)
    aligned_ret = port_ret.reindex(ep_dates).ffill().bfill().fillna(0.0)

    growth = (1.0 + aligned_ret).cumprod() * float(init_balance)
    return growth.to_numpy()
