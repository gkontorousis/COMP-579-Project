import argparse
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = "data/"

# from the original files preprocessing pipeline they used this value so we reuse it here
TICKER_COUNT_MAGIC_NUM= 4711

def load_prices(path):
    return pd.read_csv(path)


def build_daily_frames(df: pd.DataFrame, mode: str ="train"):
    data_1=df.copy()
    equal_4711_list = list(data_1.tic.value_counts() == 4711)
    names = data_1.tic.value_counts().index

    # select_stocks_list = ['NKE','KO']
    select_stocks_list = list(names[equal_4711_list])+['NKE','KO']

    data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]

    data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]

    data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

    if mode == "train":
        data = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
        daily_data = []
        for date in np.unique(data.datadate):
            daily_data.append(data[data.datadate == date])
    else:
        data = data_3[data_3.datadate > 20160000]
        daily_data = []
        for date in np.unique(data.datadate):
            daily_data.append(data[data.datadate == date])
    
    return daily_data


def sanity_check() -> None:
    parser = argparse.ArgumentParser(description="Load price CSV and build daily frames (smoke / debug).")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to dow_jones_30_daily_price.csv",
    )
    args = parser.parse_args()
    df = load_prices(args.path)
    train = build_daily_frames(df, mode="train")
    test = build_daily_frames(df, mode="test")
    print("train days:", len(train), "test days:", len(test))
    assert len(train) > 0 and len(test) > 0
    assert all(len(d) == 28 for d in train)  # only if 28 is always true for your data


if __name__ == "__main__":
    sanity_check()




# ORIGINAL METHODS

def __data_preprocess_test(df):
    data_1=df.copy()
    equal_4711_list = list(data_1.tic.value_counts() == 4711)
    names = data_1.tic.value_counts().index

    # select_stocks_list = ['NKE','KO']
    select_stocks_list = list(names[equal_4711_list])+['NKE','KO']

    data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]

    data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]

    data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

    test_data = data_3[data_3.datadate > 20160000]
    test_daily_data = []
    for date in np.unique(test_data.datadate):
        test_daily_data.append(test_data[test_data.datadate == date])

    return test_daily_data


def __data_preprocess_train(df):
    data_1=df.copy()
    equal_4711_list = list(data_1.tic.value_counts() == 4711)
    names = data_1.tic.value_counts().index

    # select_stocks_list = ['NKE','KO']
    select_stocks_list = list(names[equal_4711_list])+['NKE','KO']

    data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]

    data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]

    data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

    train_data = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
    train_daily_data = []
    for date in np.unique(train_data.datadate):
        train_daily_data.append(train_data[train_data.datadate == date])


    return train_daily_data