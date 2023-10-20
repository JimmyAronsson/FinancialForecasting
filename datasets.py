import os
import posixpath

import torch
from torch.utils.data import Dataset

from datatypes import Stock
from debug import Debug
from preprocessing import log_normalize


class DatasetLSTM(Dataset):
    def __init__(self, data_dir, filelist, forecast_steps=1, start_date='2000-01-01', final_date='2020-01-01', **kwargs):
        """
        Creates data of shape [N, L, Ch] where
           * N = Batch/dataset size
           * L = Length,
           * Ch = Channels
        """
        self.data_dir = data_dir
        self.filelist = filelist
        self.steps = forecast_steps
        self.start_date = start_date
        self.final_date = final_date

        self.stock_values = []
        self.trend_values = []

    @staticmethod
    def _save(initval, ticker):
        initval = initval[["open", "high", "low", "close"]].values
        print(initval)
        with open("./data/initvals.txt", "r+") as file:
            needle = f'{ticker}:\t{initval}\n'
            for line in file:  # Search if initval already exists in file
                if needle in line:
                    break
            else:  # not found, we are at the eof
                file.write(needle)  # append missing data

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, stock_id):

        path = posixpath.join(self.data_dir, self.filelist[stock_id])
        stock = Stock(path=path)

        stock.set_date_range(start_date=self.start_date, final_date=self.final_date)
        initval, stock.df = log_normalize(stock.df)

        # stock.detrend()

        # FIXME: Cannot use detrend BEFORE log_normalize as detrend can make values negative.
        #        Cannot use detrend AFTER log_normalize because we cannot as easily denormalize.

        time_series = torch.tensor(stock.df.values, dtype=torch.float32)

        # trend = ...

        return time_series

    def get_item(self, stock_id, inputs_and_labels=False):
        time_series = self.__getitem__(stock_id)

        if inputs_and_labels:
            inputs = time_series[:-self.steps, :]
            label = time_series[-self.steps:, :]
            return inputs, label
        else:
            return time_series
