from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# Major x-axis ticks: first trading day on or after start + k * this many months (then last episode date).
PLOT_X_TICK_STEP_MONTHS = 6


def episode_dates_from_daily_data(daily_data: list[Any]) -> pd.DatetimeIndex:
    """Trading dates for each row of `daily_data` (same length as `asset_memory` when episode completes)."""
    ints = pd.Series([day_df["datadate"].iloc[0] for day_df in daily_data])
    return pd.to_datetime(ints.astype(str), format="%Y%m%d", errors="coerce")


def episode_xaxis_tick_mdates(dates: pd.DatetimeIndex, *, step_months: int) -> np.ndarray:
    """Matplotlib date numbers for ticks on real episode trading days (~`step_months` apart)."""
    dates = pd.DatetimeIndex(dates).sort_values()
    if len(dates) == 0:
        return np.array([], dtype=float)
    ticks: list[pd.Timestamp] = [pd.Timestamp(dates[0])]
    k = 1
    while True:
        target = dates[0] + pd.DateOffset(months=step_months * k)
        if target > dates[-1]:
            break
        i = int(dates.searchsorted(target, side="left"))
        if i < len(dates):
            t = pd.Timestamp(dates[i])
            if t != ticks[-1]:
                ticks.append(t)
        k += 1
    last = pd.Timestamp(dates[-1])
    if ticks[-1] != last:
        ticks.append(last)
    return mdates.date2num(pd.DatetimeIndex(ticks))


def save_episode_result_figure(env: Any, *, outfile: str | Path | None = None) -> Path:
    """Plot agent curve and (in test mode) benchmarks vs episode dates; save PNG next to cwd unless `outfile` is set."""
    dates = episode_dates_from_daily_data(env.daily_data)
    mode = env.mode
    if outfile is None:
        outfile = "result_test.png" if mode == "test" else "result_training.png"
    outfile = Path(outfile)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, env.asset_memory, "r", label="agent")
    if mode == "test":
        ax.plot(dates, env.dji_growth, label="DJIA")
        ax.plot(dates, env.min_variance_growth, label="Min-Var")

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    tick_mdates = episode_xaxis_tick_mdates(dates, step_months=PLOT_X_TICK_STEP_MONTHS)
    ax.xaxis.set_major_locator(FixedLocator(tick_mdates))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    return outfile
