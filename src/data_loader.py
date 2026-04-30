import argparse
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = "data/"

# from the original files preprocessing pipeline they used this value so we reuse it here
TICKER_COUNT_MAGIC_NUM = 4711


def load_prices(path):
    return pd.read_csv(path)


def _build_from_original_data(df, mode: str = "train"):
    if mode not in ["train", "validation", "test"]:
        raise ValueError("Invalid mode")

    data_1 = df.copy()
    equal_magic_num_list = list(data_1.tic.value_counts() == TICKER_COUNT_MAGIC_NUM)
    names = data_1.tic.value_counts().index

    # select_stocks_list = ['NKE','KO']
    select_stocks_list = list(names[equal_magic_num_list]) + ["NKE", "KO"]

    data_2 = data_1[data_1.tic.isin(select_stocks_list)][
        ~data_1.datadate.isin(["20010912", "20010913"])
    ]

    data_3 = data_2[["iid", "datadate", "tic", "prccd", "ajexdi"]]

    data_3["adjcp"] = data_3["prccd"] / data_3["ajexdi"]

    # note here that we use datadates corresponding to the paper's specification of the train/validation/test splits
    # which are interestingly different from the original implementation
    if mode == "train":
        data = data_3[(data_3.datadate >= 20090101) & (data_3.datadate <= 20141231)]
        daily_data = []
        for date in np.unique(data.datadate):
            daily_data.append(data[data.datadate == date])
    elif mode == "validation":
        data = data_3[(data_3.datadate >= 20150101) & (data_3.datadate < 20160101)]
        daily_data = []
        for date in np.unique(data.datadate):
            daily_data.append(data[data.datadate == date])
    else:
        data = data_3[(data_3.datadate >= 20160101) & (data_3.datadate < 20181001)]
        daily_data = []
        for date in np.unique(data.datadate):
            daily_data.append(data[data.datadate == date])

    return daily_data


def _build_from_yfinance(df):
    daily_data = []
    for date in np.unique(df.datadate):
        daily_data.append(df[df.datadate == date])

    return daily_data


def build_daily_frames(df: pd.DataFrame, mode: str = "train", from_yfinance=False):
    if not from_yfinance:
        return _build_from_original_data(df, mode)
    else:
        return _build_from_yfinance(df)


def sanity_check() -> None:
    parser = argparse.ArgumentParser(
        description="Load price CSV and build daily frames (smoke / debug)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to dow_jones_30_daily_price.csv",
    )
    args = parser.parse_args()
    df = load_prices(args.path)
    train = build_daily_frames(df, mode="train")
    validation = build_daily_frames(df, mode="validation")
    test = build_daily_frames(df, mode="test")
    print("train days:", len(train), "validation days:", len(validation), "test days:", len(test))
    assert len(train) > 0 and len(validation) > 0 and len(test) > 0
    assert (
        all(len(d) == 28 for d in train)
        and all(len(d) == 28 for d in validation)
        and all(len(d) == 28 for d in test)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load price CSV and build daily frames (smoke / debug)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to dow_jones_30_daily_price.csv",
    )
    args = parser.parse_args()

    sanity_check()
