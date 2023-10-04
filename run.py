import os
import random
import posixpath
import pandas as pd
from datetime import datetime

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

logger = TensorBoardLogger("tb_logs", name="my_model")


# TODO:
#  1. Make randomized train/val/test split.
#  2. Use model.train() and model.eval() methods.


# TODO: Refactor into models module
class FinancialForecaster(pl.LightningModule):
    def __init__(self, model, data_dir, tb_logger=None, batch_size=1, prediction_steps=12, start_date=None,
                 debug=False):
        super().__init__()

        self.tb_logger = tb_logger

        self._debug_status(debug)

        self.start_date = start_date

        self.steps = prediction_steps
        self.batch_size = batch_size

        self.model = model

        self.criterion = MSELoss()

        # TODO: Remove much of the below. Replace with proper logging.

        self.data_dir = data_dir
        self.train_filelist, self.val_filelist = self._train_val_split()

        self._log_files()

        self.loss = None

        self.train_pred = None
        self.train_labels = None
        self.train_loss = []

        self.val_pred = None
        self.val_labels = None
        self.val_loss = []

        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()

    @staticmethod
    def _debug_status(debug):
        Debug.set_status(debug)
        torch.set_printoptions(threshold=200)  # Threshold for summarizing tensors rather than full repr.

    def _log_files(self):
        if self.tb_logger is None:
            return

        log_dir = self.tb_logger.log_dir
        os.mkdir(log_dir)
        with open(os.path.join(log_dir, 'data_dir.txt'), 'w') as f:
            f.write(self.data_dir)
        with open(os.path.join(log_dir, 'train_filelist.txt'), 'w') as f:
            for file in self.train_filelist:
                f.write(file + '\n')
        with open(os.path.join(log_dir, 'val_filelist.txt'), 'w') as f:
            for file in self.val_filelist:
                f.write(file + '\n')

    def _save_path(self):
        """
        Create save path for the model.
        Example: 2023-08-23--12-09_NASDAQ-small
        """

        # TODO: Add model information. (Possibly in txt-file inside dir.)
        time = datetime.now().strftime("%Y-%m-%d--%H-%M")
        data = self.data_dir.split('/')[1]
        run_info = time + '_' + data
        return posixpath.join('results/', run_info)

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

    def forward(self, x, labels=None):

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
                                    prediction_steps=self.steps,
                                    start_date=self.start_date)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["inputs"], batch["labels"]

        outputs = self.forward(inputs, labels)
        preds = outputs[:, -self.steps:, :]

        Debug.print(f"[training_step] labels: {labels}")
        Debug.print(f"[training_step] pred: {preds}")

        loss = self.criterion(preds, labels)
        self.log('train_loss', loss, on_epoch=True)

        self.train_pred = preds.detach().numpy()
        self.train_labels = labels.detach().numpy()

        # self.train_acc(pred, labels)

        Debug.print(f"[training_step] size of inputs when loaded: {inputs.size()}")
        Debug.print(f"[training_step] size of labels when loaded: {labels.size()}")
        Debug.print(f"[training_step] size of outputs: {outputs.size()}\n")

        return {"loss": loss}

    def val_dataloader(self):
        val_dataset = DatasetLSTM(data_dir=self.data_dir,
                                  filelist=self.val_filelist,
                                  prediction_steps=self.steps,
                                  start_date=self.start_date)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["inputs"], batch["labels"]

        outputs = self.forward(inputs, labels)
        preds = outputs[:, -self.steps:, :]

        Debug.print(f"[validation_step] labels: {labels}")
        Debug.print(f"[validation_step] pred: {preds}")

        loss = self.criterion(preds, labels)
        self.log('val_loss', loss, on_epoch=True)

        self.val_pred = preds.detach().numpy()
        self.val_labels = labels.detach().numpy()

        # self.val_acc(pred, labels)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters())
        return optimizer


def main():
    tb_logger = TensorBoardLogger(save_dir="logs", name="LSTM")

    model = ModelLSTM(batch_size=4,
                      nlayers=10,
                      input_size=4,
                      hidden_size=20)

    model = FinancialForecaster(model=model,
                                data_dir='data/monthly/NASDAQ-small/',
                                start_date='2000-01-01',
                                tb_logger=tb_logger,
                                batch_size=4,
                                debug=False,
                                )

    trainer = pl.Trainer(accelerator='cpu',
                         callbacks=[CallbacksLSTM()],
                         log_every_n_steps=1,
                         logger=tb_logger,
                         max_epochs=200,
                         )

    trainer.fit(model, ckpt_path=None)


if __name__ == "__main__":
    main()
