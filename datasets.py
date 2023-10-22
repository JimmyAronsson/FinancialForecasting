import posixpath
import torch
from torch.utils.data import Dataset
from typing import Literal

from configs import Config
from datatypes import Stock
from preprocessing import log_normalize


class DatasetLSTM(Dataset):
    def __init__(self, config: Config, stage: Literal['train', 'val']):
        self.config = config

        self.filelist = config.filelist[stage]

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

        path = posixpath.join(self.config.data_dir, self.filelist[stock_id])
        stock = Stock(path=path)

        stock.set_date_range(start_date=self.config.time_period[0], final_date=self.config.time_period[1])
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
            inputs = time_series[:-self.config.forecast_steps, :]
            label = time_series[-self.config.forecast_steps:, :]
            return inputs, label
        else:
            return time_series
