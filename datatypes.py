import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


class Stock:
    def __init__(self, path):
        self.df_initval = None
        self.path = path
        self.ticker = self._ticker()
        self.df = self._load_data()
        self.date_range = self._date_range()

        self.trend = None  # Trend component of data, see self.detrend()

    def detrend(self, method='ma', res='daily'):
        """
        Detrend time series using one of four methods:
        * 'constant' - Assumes constant trend
        * 'linear' - Assumes linear trend
        * 'quadratic' - Assumes quadratic trend
        * 'ma' (default) - Moving average for non-stationary data
        """
        self.trend = pd.DataFrame()
        if method == 'ma':
            time_resolution = {
                'daily': 1600 * (365 / 4) ** 4,
                'weekly': 1600 * (52 / 4) ** 4,
                'monthly': 1600 * (12 / 4) ** 4,
                'quarterly': 1600 * (4 / 4) ** 4,
                'yearly': 1600 * (1 / 4) ** 4
            }
            # Hodrick-Prescott filter
            channels = ['open', 'high', 'low', 'close']
            for ch in channels:
                self.df[ch], self.trend[ch] = hpfilter(self.df[ch], lamb=time_resolution[res])

    def _date_range(self):
        return self.df.index[0], self.df.index[-1]

    def _ticker(self):
        return self.path[:-4].split('/')[-1].split('\\')[-1]  # Path is of the form '.../filename.csv'

    def _load_data(self):
        df = pd.read_csv(self.path, index_col="date", parse_dates=["date"], na_filter=False,
                         usecols=lambda x: x != "ticker")
        df = df.replace(to_replace=0, value=np.nan).ffill()  # Interpolate missing values.

        return df

    def set_start_date(self, start_date=None):
        try:
            self.df = self.df[self.df.index >= pd.to_datetime(start_date)]
        except Exception as e:
            raise e

    def uniform_timespan(self):
        with open('data/oldest_date.txt') as f:
            oldest_date = f.read()
        with open('data/most_recent_date.txt') as f:
            most_recent_date = f.read()

        # Add missing business days and interpolate values.
        bdate_range = pd.bdate_range(oldest_date, most_recent_date)  # business days range
        self.df = self.df.reindex(bdate_range)
        self.df.interpolate(method='time', inplace=True)  # Interpolate missing values in middle
        self.df.fillna(1, inplace=True)  # Fill missing values at beginning. Becomes 0 after log_normalization.

    def get_date_range(self):
        return self.date_range

    def get_ticker(self):
        return self.ticker

    def log_normalize(self):  # TODO: Refactor to preprocessing module
        """
        Given a time series p(t), computes its log-normalization
        y(t) = log( p(t) / p(t-1) )

        and the initial value

        y0 = log( p(0) ).
        """

        self.df_initval = np.log(self.df.head(1))
        self.df = (self.df / self.df.shift(periods=1)).fillna(1).apply(lambda x: np.log(x))

    def log_denormalize(self):  # TODO: Refactor to preprocessing module
        """
        Returns a time series P(t) from its log-normalization

        y(t) = log( P(t) / P(t-1) )

        and its initial value

        y0 = log( P(0) ).
        """
        log_df = self.df_initval.values + self.df.cumsum()
        self.df = log_df.apply(lambda x: np.exp(x))
