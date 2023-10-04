import os
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


def main():
    """
    resample_data('B')
    resample_data('W')
    resample_data('M')
    """


if __name__ == "__main__":
    main()
