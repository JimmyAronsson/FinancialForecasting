import os
import random
import posixpath
import pandas as pd
from datetime import datetime

from icecream import ic

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from debug import Debug
from datasets import DatasetLSTM
from callbacks import CallbacksLSTM
from models import ModelLSTM

from pytorch_lightning.loggers import TensorBoardLogger

from loggers import LSTMLogger

# TODO:
#  1. Make randomized train/val/test split.
#  2. Use model.train() and model.eval() methods.


# TODO: Refactor into models module
class FinancialForecaster(pl.LightningModule):
    def __init__(self, model, data_dir, batch_size=1, forecast_steps=12,
                 start_date=None, final_date=None, train_split=0.8, debug=False, **kwargs):
        super().__init__()

        self._debug_status(debug)

        self.start_date = start_date
        self.final_date = final_date

        self.steps = forecast_steps
        self.batch_size = batch_size

        self.model = model

        self.criterion = MSELoss()

        # TODO: Remove much of the below. Replace with proper logging.

        self.data_dir = data_dir
        self.train_filelist, self.val_filelist = self._train_val_split(train_split)

        self.loss = None
        self.train_loss = []
        self.val_loss = []

        self.save_hyperparameters()

        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()

    @staticmethod
    def _debug_status(debug):
        Debug.set_status(debug)
        torch.set_printoptions(threshold=200)  # Threshold for summarizing tensors rather than full repr.

    def _train_val_split(self, train_split=0.8):
        data_filelist = os.listdir(self.data_dir)
        ntrain = round(train_split * len(data_filelist))

        random.shuffle(data_filelist)
        train_filelist = data_filelist[:ntrain]
        val_filelist = data_filelist[ntrain:]

        print("Number of training files: ", len(train_filelist))
        print("Number of validation files: ", len(val_filelist))
        print("Number of total files: ", len(data_filelist))

        return train_filelist, val_filelist

    def forward(self, x):
        x = x[0]

        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        # Use previous day's values as initial prediction.
        init_pred = x[:, -1, :].unsqueeze(1)
        init_pred = init_pred.expand(-1, self.steps, -1)
        # Add initial prediction to x
        x = torch.cat((x, init_pred), dim=1)

        Debug.print(f"[forward] size of x after cat: {x.size()}")  # [N, L, Ch]

        return self.model.forward(x)

    def train_dataloader(self):
        train_dataset = DatasetLSTM(data_dir=self.data_dir,
                                    filelist=self.train_filelist,
                                    forecast_steps=self.steps,
                                    start_date=self.start_date,
                                    final_date=self.final_date)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        ground_truth = batch

        inputs = ground_truth[:, :-self.steps, :]
        labels = ground_truth[:, -self.steps:, :]

        forecast = self.forward(inputs)
        forecast_future = forecast[:, -self.steps:, :]

        Debug.print(f"[training_step] labels: {labels}")
        Debug.print(f"[training_step] forecast_future: {forecast_future}")

        loss = self.criterion(ground_truth, forecast)
        # loss = self.criterion(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        Debug.print(f"[training_step] size of inputs when loaded: {inputs.size()}")
        Debug.print(f"[training_step] size of labels when loaded: {labels.size()}")
        Debug.print(f"[training_step] size of forecast: {forecast.size()}\n")

        return {"loss": loss}

    def val_dataloader(self):
        val_dataset = DatasetLSTM(data_dir=self.data_dir,
                                  filelist=self.val_filelist,
                                  forecast_steps=self.steps,
                                  start_date=self.start_date,
                                  final_date=self.final_date)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)

    def validation_step(self, batch, batch_idx):
        ground_truth = batch  # Historic and future data

        inputs = ground_truth[:, :-self.steps, :]  # Historic
        labels = ground_truth[:, -self.steps:, :]  # Future

        forecast = self.forward(inputs)  # Historic and future forecast
        forecast_future = forecast[:, -self.steps:, :]

        Debug.print(f"[validation_step] labels: {labels}")
        Debug.print(f"[validation_step] forecast_future: {forecast_future}")

        loss = self.criterion(ground_truth, forecast)
        # loss = self.criterion(preds, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return batch, self(batch)

    def log_run_info(self, log_dir):
        with open(os.path.join(log_dir, 'run_info.txt'), 'w') as f:
            f.write(f'data_dir:\t{self.data_dir}')

            f.write(f'\nstart_date:\t{self.start_date}')
            f.write(f'\nfinal_date:\t{self.final_date}')
            f.write(f'\nforecast_steps:\t{self.steps}')
            f.write(f'\nbatch_size:\t{self.batch_size}')

            f.write(f'\ntrain_filelist:\t{self.train_filelist}')
            f.write(f'\nval_filelist:\t{self.val_filelist}')


def main():
    logger = LSTMLogger(save_dir="logs", name="LSTM")
    logger.make_log_dir()

    model = ModelLSTM()

    forecaster = FinancialForecaster(model=model,
                                     data_dir='data/monthly/NASDAQ-tiny/',
                                     train_split=0.5,
                                     start_date='2021-01-01',
                                     final_date='2023-01-01',
                                     forecast_steps=5,
                                     batch_size=1,
                                     debug=False,
                                     )

    trainer = pl.Trainer(accelerator='cpu',
                         callbacks=[CallbacksLSTM()],
                         logger=logger,
                         log_every_n_steps=20,
                         max_epochs=3000,
                         )
    forecaster.log_run_info(logger.log_dir)
    trainer.fit(forecaster, ckpt_path=None)
    

if __name__ == "__main__":
    main()
