import os
import posixpath
import pandas as pd
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import DatasetLSTM

from debug import Debug

# TODO: Make randomized train/val/test split.


# TODO: Refactor into models module
class ModelLSTM(pl.LightningModule):
    def __init__(self, data_dir, batch_size=1, pred_ndays=100, start_date=None, debug=False):
        super().__init__()

        self._debug_status(debug)

        self.start_date = start_date

        self.pred_ndays = pred_ndays
        self.batch_size = batch_size
        self.hidden = None  # TODO: Create _load_parameters() method

        self.input_size = 4
        self.hidden_size = 20
        self.num_layers = 10
        self.LSTM = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            # proj_size=self.input_size,
                            # dropout=0.3,
                            )
        self.linear = nn.Linear(self.hidden_size, 4)
        self.conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=4, track_running_stats=False)
        self.criterion = MSELoss()

        # TODO: Remove much of the below. Replace with proper logging.

        self.data_dir = data_dir
        self.loss = None

        self.train_dir = posixpath.join(data_dir, 'train')
        self.train_date_range = (None, None)
        self.train_tickers = []
        self.train_pred = None
        self.train_labels = None
        self.train_loss = []

        self.val_dir = posixpath.join(data_dir, 'val')
        self.val_date_range = (None, None)
        self.val_tickers = []
        self.val_pred = None
        self.val_labels = None
        self.val_loss = []

        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()

    @staticmethod
    def _debug_status(debug):
        Debug.set_status(debug)
        torch.set_printoptions(threshold=200)  # Threshold for summarizing tensors rather than full repr.

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

    def forward(self, x, labels=None):
        hidden = (torch.rand(self.num_layers, self.batch_size, self.hidden_size, requires_grad=True),
                  torch.rand(self.num_layers, self.batch_size, self.hidden_size, requires_grad=True))

        # Use previous day's values as initial prediction.
        init_pred = x[:, -1, :].unsqueeze(1)
        init_pred = init_pred.expand(-1, self.pred_ndays, -1)

        # Add initial prediction to x
        x = torch.cat((x, init_pred), dim=1)
        Debug.print(f"[forward] size of x after cat: {x.size()}")  # [N, L, Ch]

        x, self.hidden = self.LSTM(x, hidden)
        x = self.activation(x)
        x = self.linear(x)

        pred = x[:, -self.pred_ndays:, :]

        Debug.print(f"[forward] labels: {labels}")
        Debug.print(f"[forward] pred: {pred}")

        if labels is not None:
            self.loss = self.criterion(pred, labels)
        return self.loss, x

    def train_dataloader(self):
        config = {'data_dir': self.train_dir,
                  'pred_ndays': self.pred_ndays}
        train_dataset = DatasetLSTM(**config)
        self.train_tickers = train_dataset.get_tickers()
        self.train_date_range = train_dataset.get_date_range()
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["inputs"], batch["labels"]

        loss, outputs = self.forward(inputs, labels)
        self.log('train_loss', loss, on_epoch=True)

        pred = outputs[:, -self.pred_ndays:, :]
        self.train_pred = pred.detach().numpy()
        self.train_labels = labels.detach().numpy()

        # self.train_acc(pred, labels)

        Debug.print(f"[training_step] size of inputs when loaded: {inputs.size()}")
        Debug.print(f"[training_step] size of labels when loaded: {labels.size()}")
        Debug.print(f"[training_step] size of outputs: {outputs.size()}\n")

        print(f" train_loss: {loss.item()}")  # Always print.

        self.train_loss.append(loss.item())

        return {"loss": loss}

    def val_dataloader(self):
        config = {'data_dir': self.val_dir,
                  'pred_ndays': self.pred_ndays}
        val_dataset = DatasetLSTM(**config)
        self.val_tickers = val_dataset.get_tickers()
        self.val_date_range = val_dataset.get_date_range()
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["inputs"], batch["labels"]

        loss, outputs = self.forward(inputs, labels)
        self.log('val_loss', loss, on_epoch=True)

        pred = outputs[:, -self.pred_ndays:, :]
        self.val_pred = pred.detach().numpy()
        self.val_labels = labels.detach().numpy()

        # self.val_acc(pred, labels)

        print(f" val_loss: {loss.item()}")

        self.val_loss.append(loss.item())

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer

    # TODO: Refactor into support class
    def save(self):
        """
        Save the model.
        """
        save_path = self._save_path()
        os.mkdir(save_path)

        # Save model parameters and losses
        torch.save(self.state_dict(), posixpath.join(save_path, 'model.pth'))

        train_loss = pd.DataFrame(self.train_loss, columns=["loss"])
        train_loss.to_csv(posixpath.join(save_path, "train_loss.csv"))

        val_loss = pd.DataFrame(self.val_loss, columns=["loss"])
        val_loss.to_csv(posixpath.join(save_path, "val_loss.csv"))

        # Save train results
        train_path = posixpath.join(save_path, 'train/')
        os.mkdir(train_path)

        # FIXME: Fix this. len(train_dates) = FULL length of data, should only be last npred_days.
        train_dates = pd.bdate_range(start=self.train_date_range[0],
                                     end=self.train_date_range[1])
        train_dates = train_dates[-self.pred_ndays:]

        print("Length of train_dates: ", len(train_dates))
        print("Length of train_pred[i]: ", len(self.train_pred[0]))
        for i in range(len(self.train_tickers)):
            pred = pd.DataFrame(self.train_pred[i], columns=["open", "high", "low", "close"])
            pred.insert(loc=0, column="date", value=train_dates)
            pred.to_csv(posixpath.join(train_path, f'train_pred_{self.train_tickers[i]}.csv'), index=False)

            labels = pd.DataFrame(self.train_labels[i], columns=["open", "high", "low", "close"])
            labels.insert(loc=0, column="date", value=train_dates)
            labels.to_csv(posixpath.join(train_path, f'train_labels_{self.train_tickers[i]}.csv'), index=False)

        # Save val results
        val_path = posixpath.join(save_path, 'val/')
        os.mkdir(val_path)

        # FIXME: Fix this. len(val_dates) = FULL length of data, should only be last npred_days.
        val_dates = pd.bdate_range(start=self.val_date_range[0],
                                   end=self.val_date_range[1])
        val_dates = val_dates[-self.pred_ndays:]

        for i in range(len(self.val_tickers)):
            pred = pd.DataFrame(self.val_pred[i], columns=["open", "high", "low", "close"])
            pred.insert(loc=0, column="date", value=val_dates)
            pred.to_csv(posixpath.join(val_path, f'val_pred_{self.val_tickers[i]}.csv'), index=False)

            labels = pd.DataFrame(self.val_labels[i], columns=["open", "high", "low", "close"])
            labels.insert(loc=0, column="date", value=val_dates)
            labels.to_csv(posixpath.join(val_path, f'val_labels_{self.val_tickers[i]}.csv'), index=False)

        print("Finished saving model at:", save_path[2:])  # Print path without '..'


def main():
    model = ModelLSTM(data_dir='data/NASDAQ-small/', batch_size=4, debug=False)
    trainer = pl.Trainer(max_epochs=1, accelerator='cpu', log_every_n_steps=1)
    trainer.fit(model)
    model.save()


if __name__ == "__main__":
    main()
