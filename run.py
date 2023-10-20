import os
import random
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models import ModelLSTM
from loggers import LSTMLogger
from datasets import DatasetLSTM
from callbacks import CallbacksLSTM


class FinancialForecaster(pl.LightningModule):
    def __init__(self, model, data_dir, batch_size=1, forecast_steps=12,
                 start_date=None, final_date=None, train_split=0.8, **kwargs):
        super().__init__()

        # TODO: Refactor dates, steps, etc. into config data structure

        self.start_date = start_date
        self.final_date = final_date

        self.steps = forecast_steps
        self.batch_size = batch_size

        self.data_dir = data_dir
        # TODO: The below should not be run e.g. in EvaluateForecaster.__init__(). Refactor?
        self.train_filelist, self.val_filelist = self._train_val_split(train_split)

        self.model = model
        self.criterion = MSELoss()  # TODO: Refactor?

        self.save_hyperparameters()

    def _train_val_split(self, train_split=0.8):
        data_filelist = os.listdir(self.data_dir)
        ntrain = round(train_split * len(data_filelist))

        random.shuffle(data_filelist)
        train_filelist = data_filelist[:ntrain]
        val_filelist = data_filelist[ntrain:]

        return train_filelist, val_filelist

    def forward(self, x):
        if x.dim() == 2:  # If unbached
            x = x.unsqueeze(0)

        # Use last day's values as initial future forecast and add to x
        x_forward_steps = x[:, -1, :].unsqueeze(1).expand(-1, self.steps, -1)
        x = torch.cat((x, x_forward_steps), dim=1)

        forecast = self.model.forward(x)

        return forecast

    def train_dataloader(self):
        train_dataset = DatasetLSTM(data_dir=self.data_dir,
                                    filelist=self.train_filelist,
                                    forecast_steps=self.steps,
                                    start_date=self.start_date,
                                    final_date=self.final_date)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        ground_truth = batch  # Ground truth (historic and forwards)
        gt_historic = ground_truth[:, :-self.steps, :]

        forecast = self.forward(gt_historic)

        loss = self.criterion(ground_truth, forecast)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss}

    def val_dataloader(self):
        val_dataset = DatasetLSTM(data_dir=self.data_dir,
                                  filelist=self.val_filelist,
                                  forecast_steps=self.steps,
                                  start_date=self.start_date,
                                  final_date=self.final_date)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)

    def validation_step(self, batch, batch_idx):
        ground_truth = batch  # Ground truth (historic and forwards)
        gt_historic = ground_truth[:, :-self.steps, :]

        forecast = self.forward(gt_historic)

        loss = self.criterion(ground_truth, forecast)
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

    forecaster = FinancialForecaster(model=ModelLSTM(),
                                     data_dir='data/monthly/NASDAQ-tiny/',
                                     train_split=0.5,
                                     start_date='2021-01-01',
                                     final_date='2023-01-01',
                                     forecast_steps=5,
                                     batch_size=1,
                                     )

    trainer = pl.Trainer(accelerator='cpu',
                         callbacks=[CallbacksLSTM()],
                         logger=logger,
                         max_epochs=3000,
                         )

    forecaster.log_run_info(logger.log_dir)

    trainer.fit(forecaster, ckpt_path=None)
    

if __name__ == "__main__":
    main()
