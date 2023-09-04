import os
import numpy as np
import pandas as pd

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# TODO: Refactor log_normalize and log_denormalize from Stock class to here.


class Transform:

    @staticmethod
    def hpfilter(df, res='daily'):
        """Hodrick-Prescott filter"""
        time_resolution = {
            'daily': 1600 * (365 / 4) ** 4,
            'weekly': 1600 * (52 / 4) ** 4,
            'monthly': 1600 * (12 / 4) ** 4,
            'quarterly': 1600 * (4 / 4) ** 4,
            'yearly': 1600 * (1 / 4) ** 4
        }
        cycle, trend = hpfilter(df, lamb=time_resolution[res])
        return cycle, trend

    @staticmethod
    def seasonal_decompose(df, period=365, model='multiplicative'):
        return seasonal_decompose(df, period=period, model=model)

    @staticmethod
    def ewma(df, com=None, span=None, halflife=None):
        """Exponentially weighted moving average"""
        return df.ewm(com, span, halflife)

    # TODO: Refactor. exp_smoothing is more forecasting than detrending.
    @staticmethod
    def exp_smoothing(df, trend='multiplicative'):
        """Holt-Winters (triple) exponential smoothing"""
        return ExponentialSmoothing(df, trend=trend, seasonal=None, seasonal_periods=None, freq=None)
