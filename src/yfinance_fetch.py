from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import yfinance as yf

from datetime import datetime


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch prices from Yahoo Finance.")
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--tickers_path",
        type=Path,
        help="Path to tickers file that contains the tickers to fetch information for.",
    )
    parser.add_argument(
        "--date_start",
        type=parse_date,
        required=True,
        help="The start date for fetching data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--date_end",
        type=parse_date,
        required=True,
        help="The end date for fetching data in YYYY-MM-DD format.",
    )

    args = parser.parse_args()
    tickers = [line.strip() for line in args.tickers_path.read_text().splitlines()]
    print(f"Loaded {len(tickers)} tickers from {args.tickers_path}")

    start_date = args.date_start
    end_date = args.date_end
    print("Date window:", start_date, "to", end_date)
    raw = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
    )

    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise SystemExit("Yahoo returned nothing (check dates / tickers / network).")

    close_wide = raw.xs("Close", axis=1, level=1)
    long_df = close_wide.stack().reset_index()
    long_df.columns = ["date", "tic", "adjcp"]
    long_df["datadate"] = long_df["date"].dt.strftime("%Y%m%d").astype(int)
    long_df["adjcp"] = pd.to_numeric(long_df["adjcp"], errors="coerce")
    long_df = long_df.dropna(subset=["adjcp"])
    long_df = long_df.drop_duplicates(["datadate", "tic"], keep="last")
    pv = long_df.pivot(index="datadate", columns="tic", values="adjcp").dropna(how="any")
    if pv.empty:
        raise SystemExit("No day has all tickers; widen dates or fix symbols.")
    good_days = set(pv.index.astype(int))
    out = long_df[long_df["datadate"].isin(good_days)].sort_values(["datadate", "tic"])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Add prefix to output_path with start and end date in YYYYMMDD_YYYYMMDD
    date_prefix = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_"
    output_path_with_prefix = args.output_path.parent / (date_prefix + args.output_path.name)
    out.to_csv(output_path_with_prefix, index=False)
    print(
        f"Saved {len(out):,} rows → {output_path_with_prefix} ({len(good_days)} days, {pv.shape[1]} tickers)"
    )
