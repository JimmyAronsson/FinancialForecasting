import os
import pytorch_lightning as pl
import random
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import DatasetLSTM


class FinancialForecaster(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = config.get_model()
        self.criterion = MSELoss()  # TODO: Refactor?
        self.save_hyperparameters()

    def forward(self, x):
        if x.dim() == 2:  # If unbached
            x = x.unsqueeze(0)

        # Use last day's values as initial future forecast and add to x
        x_forward_steps = x[:, -1, :].unsqueeze(1).expand(-1, self.config.forecast_steps, -1)
        x = torch.cat((x, x_forward_steps), dim=1)

        forecast = self.model.forward(x)

        return forecast

    def train_dataloader(self):
        train_dataset = DatasetLSTM(self.config, stage='train')
        return DataLoader(train_dataset, batch_size=self.config.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        ground_truth = batch  # Ground truth (historic and forwards)
        gt_historic = ground_truth[:, :-self.config.forecast_steps, :]

        forecast = self.forward(gt_historic)

        loss = self.criterion(ground_truth, forecast)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss}

    def val_dataloader(self):
        val_dataset = DatasetLSTM(self.config, stage='val')
        return DataLoader(val_dataset, batch_size=self.config.batch_size, num_workers=4)

    def validation_step(self, batch, batch_idx):
        ground_truth = batch  # Ground truth (historic and forwards)
        gt_historic = ground_truth[:, :-self.config.forecast_steps, :]

        forecast = self.forward(gt_historic)

        loss = self.criterion(ground_truth, forecast)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return batch, self(batch)
