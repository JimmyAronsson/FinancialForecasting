import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    """Financial forecasting base model"""

    def __init__(self):
        super().__init__()
        # self.weights = ...
        # self.biases = ...

        self.loss = None

    def forward(self, inputs, target):
        return self.model(inputs, target)


class ModelLSTM(nn.Module):
    def __init__(self, batch_size, nlayers, input_size, hidden_size):
        super().__init__()
        self.h0 = torch.zeros(nlayers, batch_size, hidden_size)
        self.c0 = torch.zeros(nlayers, batch_size, hidden_size)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=nlayers,
                            batch_first=True,
                            # proj_size=self.input_size,
                            # dropout=0.3,
                            )
        self.linear = nn.Linear(hidden_size, 4)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, (h, c) = self.lstm(x, (self.h0, self.c0))
        x = self.activation(x)
        x = self.linear(x)

        return x
