import os
import posixpath

import torch
from torch.utils.data import Dataset

from datatypes import Stock
from debug import Debug


class DatasetLSTM(Dataset):
    def __init__(self, data_dir, pred_ndays=1):
        """
        Creates data of shape [N, L, Ch] where
           * N = Batch/dataset size
           * L = Length,
           * Ch = Channels
        """
        self.data_dir = data_dir
        self.filelist = os.listdir(data_dir)
        self.pred_ndays = pred_ndays

        self.date_range = (None, None)
        self.tickers = []
        self.stock_values = []
        self.trend_values = []

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, stock_id, transform=None):

        path = posixpath.join(self.data_dir, self.filelist[stock_id])
        stock = Stock(path=path)

        stock.uniform_timespan()
        stock.set_start_date(start_date='2000-01-01')
        stock.log_normalize()
        # stock.detrend()

        # FIXME: Cannot use detrend BEFORE log_normalize as detrend can make values negative.
        #        Cannot use detrend AFTER log_normalize because we cannot as easily denormalize.

        data = torch.tensor(stock.df.values, dtype=torch.float32)
        Debug.print(f"[DatasetLSTM] Size of data after cat: {data.size()}")

        # trend = ...

        inputs = data[:-self.pred_ndays, :]
        labels = data[-self.pred_ndays:, :]

        return {"inputs": inputs, "labels": labels}

    def get_tickers(self):
        return self.tickers

    def get_date_range(self):
        return self.date_range