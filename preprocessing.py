import os

import numpy as np
import pandas as pd

from datatypes import Stock


def resample_data(freq=None):
    """Resample original NASDAQ data into daily, weekly, and monthly data."""
    if freq not in ['B', 'W', 'M']:
        print("Please choose a valid frequency:\n\
              Business daily: freq = 'B'\n\
              Weekly: freq = 'W'\n\
              Monthly: freq = 'M'")
        exit()

    base = ".\\data\\original"
    for root, dirs, files in os.walk(base):
        if len(files) == 0:
            continue

        for file in files:
            load_path = os.path.join(root, file)
            stock = Stock(load_path)
            stock.uniform_timespan()

            df_aggregate = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
            df_aggregate["open"] = stock.df["open"].resample(freq).first()
            df_aggregate["high"] = stock.df["high"].resample(freq).max()
            df_aggregate["low"] = stock.df["low"].resample(freq).min()
            df_aggregate["close"] = stock.df["close"].resample(freq).last()
            df_aggregate["date"] = df_aggregate.index

            path_freq = None
            if freq == 'B':  # Business daily data
                path_freq = "daily"
            if freq == 'M':  # Monthly data
                path_freq = "monthly"
            elif freq == 'W':  # Weekly data
                path_freq = "weekly"

            save_path = load_path.replace("original", path_freq)
            df_aggregate.to_csv(path_or_buf=save_path, index=False)


def log_normalize(p: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Given a time series p(t), computes its log-normalization
    y(t) = log( p(t) / p(t-1) )

    and the initial value

    y0 = log( p(0) ).
    """

    y0 = np.log(p.head(1))
    y = (p / p.shift(periods=1)).fillna(1).apply(lambda x: np.log(x))

    return y0, y


def log_denormalize(y0: pd.DataFrame, y: pd.DataFrame) -> (pd.DataFrame):
    """
    Returns a time series p(t) from its log-normalization

    y(t) = log( p(t) / p(t-1) )

    and its initial value

    y0 = log( p(0) ).
    """

    log_p = y0.values + y.cumsum()
    p = log_p.apply(lambda x: np.exp(x))

    return p
